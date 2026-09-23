"""Prune stale wheels from the rolling ``dev-wheels`` GitHub release.

GitHub allows at most 1,000 assets on a single release, so the rolling dev
channel has to delete old wheels periodically. The retention target is expressed
in days, but enforced as an asset count derived from the publishing rate the
caller passes in:

    threshold = assets_per_run * retention_days + persistent_assets

``assets_per_run`` comes from ``--incoming``, the number of wheels the current
run is about to upload; ``persistent_assets`` is counted from the release
itself. For the current matrix (4 platforms x 3 CPython builds, published
daily) that is ``12 * 60 + 4 = 724`` assets for a 60-day window.

Assets are pruned in whole *version groups*. All assets sharing a wheel version
(for example ``20260922.dev0``) are treated as one indivisible unit.

The policy, in priority order:

1. Assets that are not versioned wheels of the target package (notably the
   ``torch-mlir-opt-<platform>`` binaries, which are overwritten in place on
   every run and act as "latest" pointers) are never pruned. They are counted
   as ``persistent_assets`` and budgeted for.
2. The threshold is capped at ``--hard-limit`` so that a wider build matrix
   cannot silently compute a target above GitHub's 1,000-asset limit. Hitting
   the cap shortens the effective retention window and emits a warning.
3. Oldest version groups are deleted until the survivors plus the current run's
   uploads fit under the threshold. At least one version group is always
   retained.
"""

import argparse
import datetime
import os
import re
import sys

import packaging.version
import requests

GITHUB_API = "https://api.github.com"

# https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
# "Up to 1000 release assets may be associated with a single release."
GITHUB_MAX_RELEASE_ASSETS = 1000

# Ceiling the computed threshold is clamped to. Kept below
# GITHUB_MAX_RELEASE_ASSETS so a concurrent or partially-completed upload cannot
# tip the release over the hard limit.
DEFAULT_HARD_LIMIT = 990

DEFAULT_RETENTION_DAYS = 60

# Maximum supported by the GitHub REST API.
ASSETS_PER_PAGE = 100

# Guard against an unterminated pagination loop; 100 pages is 10,000 assets,
# an order of magnitude above the hard release limit.
MAX_ASSET_PAGES = 100

REQUEST_TIMEOUT_SECONDS = 30

# Sorts before any real upload timestamp, used when a malformed asset has no
# usable created_at so that it is cleaned up first.
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def escape_package_name(package_name: str) -> str:
    """Escape a package name per PEP 427 wheel filename rules."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", package_name).lower()


def parse_wheel_version(asset_name: str, escaped_package: str):
    """Return the version encoded in a wheel asset name, or None.

    Returns None for anything that is not a ``.whl`` belonging to
    ``escaped_package`` (for example the ``torch-mlir-opt-*`` binaries), which
    marks the asset as persistent and never prunable.
    """
    if not asset_name.endswith(".whl"):
        return None
    # PEP 427: {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    parts = asset_name.split("-")
    if len(parts) < 2:
        return None
    if escape_package_name(parts[0]) != escaped_package:
        return None
    return parts[1]


def parse_timestamp(value: str) -> datetime.datetime:
    """Parse a GitHub RFC 3339 timestamp into an aware UTC datetime."""
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Could not parse GitHub timestamp '{value}': {e}") from e
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


class VersionGroup:
    """All release assets that share a single wheel version."""

    def __init__(self, version: str):
        self.version = version
        self.assets = []

    def add(self, asset):
        self.assets.append(asset)

    @property
    def newest_created_at(self) -> datetime.datetime:
        """Upload time of the most recent asset in the group.

        Only used to order groups that cannot be ordered by version; the
        retention decision itself is based on asset counts, not timestamps.
        """
        stamps = [
            parse_timestamp(a["created_at"]) for a in self.assets if a.get("created_at")
        ]
        return max(stamps) if stamps else _EPOCH

    @property
    def total_bytes(self) -> int:
        return sum(a.get("size", 0) for a in self.assets)

    def sort_key(self):
        """Order groups chronologically, oldest first.

        Sorts on the parsed PEP 440 version because the dev channel uses
        date-based versions (``YYYYMMDD.devN``), making version order the
        semantically correct chronological order. Unparseable versions sort
        oldest so they are cleaned up first, and upload time breaks ties.
        """
        try:
            parsed = packaging.version.parse(self.version)
            valid = 1
        except packaging.version.InvalidVersion:
            parsed = packaging.version.parse("0")
            valid = 0
        return (valid, parsed, self.newest_created_at)


def group_assets(assets, package_name: str):
    """Split assets into version groups plus a list of persistent assets."""
    escaped_package = escape_package_name(package_name)
    groups = {}
    persistent = []
    for asset in assets:
        version = parse_wheel_version(asset.get("name", ""), escaped_package)
        if version is None:
            persistent.append(asset)
            continue
        groups.setdefault(version, VersionGroup(version)).add(asset)
    return groups, persistent


def infer_assets_per_run(groups, incoming: int) -> int:
    """Determine how many versioned wheels a single run publishes.

    Prefers the caller-supplied ``--incoming`` count. Falls back to the largest
    version group already on the release, which is what a dry run invoked
    without ``--incoming`` has to work from.
    """
    if incoming > 0:
        return incoming
    if groups:
        return max(len(g.assets) for g in groups.values())
    return 0


class PruningPlan:
    """The set of version groups selected for deletion, and why."""

    def __init__(self):
        self.doomed_versions = []
        self.assets_to_delete = []
        self.warnings = []
        self.assets_per_run = 0
        self.persistent_count = 0
        self.threshold = 0
        self.uncapped_threshold = 0
        self.effective_retention_days = 0
        self.kept_asset_count = 0
        self.projected_asset_count = 0


def compute_threshold(
    assets_per_run: int,
    retention_days: int,
    persistent_count: int,
    hard_limit: int,
):
    """Derive the asset ceiling from the observed publishing rate.

    Returns ``(threshold, uncapped_threshold)``. The two differ only when the
    requested window does not fit within ``hard_limit``.
    """
    uncapped = assets_per_run * retention_days + persistent_count
    return min(uncapped, hard_limit), uncapped


def plan_pruning(
    assets,
    package_name: str,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    incoming: int = 0,
    assets_per_run=None,
    hard_limit: int = DEFAULT_HARD_LIMIT,
) -> PruningPlan:
    """Decide which version groups to delete. Pure; performs no I/O."""
    if retention_days < 1:
        raise ValueError(f"--retention-days must be >= 1, got {retention_days}")
    if incoming < 0:
        raise ValueError(f"--incoming must be >= 0, got {incoming}")
    if hard_limit < 1:
        raise ValueError(f"--hard-limit must be >= 1, got {hard_limit}")
    if assets_per_run is not None and assets_per_run < 0:
        raise ValueError(f"--assets-per-run must be >= 0, got {assets_per_run}")

    plan = PruningPlan()
    groups, persistent = group_assets(assets, package_name)

    if assets_per_run is None:
        assets_per_run = infer_assets_per_run(groups, incoming)

    plan.assets_per_run = assets_per_run
    plan.persistent_count = len(persistent)
    plan.threshold, plan.uncapped_threshold = compute_threshold(
        assets_per_run, retention_days, len(persistent), hard_limit
    )

    if assets_per_run > 0:
        plan.effective_retention_days = (plan.threshold - len(persistent)) // (
            assets_per_run
        )
    else:
        plan.effective_retention_days = retention_days

    if plan.uncapped_threshold > hard_limit:
        plan.warnings.append(
            f"A {retention_days}-day window at {assets_per_run} assets/run needs "
            f"{plan.uncapped_threshold} assets, above the {hard_limit}-asset "
            f"limit. Capping at {hard_limit}, which is only "
            f"{plan.effective_retention_days} days of retention. Reduce "
            f"--retention-days or shrink the build matrix."
        )

    # Oldest first: this is the deletion order.
    ordered = sorted(groups.values(), key=VersionGroup.sort_key)

    doomed = set()

    def projected():
        survivors = sum(len(g.assets) for g in ordered if g.version not in doomed)
        return len(persistent) + survivors + incoming

    for group in ordered:
        if projected() <= plan.threshold:
            break
        if len(groups) - len(doomed) <= 1:
            # Never delete the last surviving version.
            break
        doomed.add(group.version)

    plan.projected_asset_count = projected()
    if plan.projected_asset_count > plan.threshold:
        plan.warnings.append(
            f"Cannot fit within the {plan.threshold}-asset threshold: "
            f"{plan.projected_asset_count} assets projected after pruning down "
            f"to the one-version safety floor ({len(persistent)} persistent "
            f"+ {plan.projected_asset_count - len(persistent) - incoming} "
            f"retained + {incoming} incoming). The threshold is too small for "
            f"a single run's output."
        )

    for group in ordered:
        if group.version in doomed:
            plan.doomed_versions.append(group.version)
            plan.assets_to_delete.extend(group.assets)

    plan.kept_asset_count = plan.projected_asset_count - incoming
    return plan


def _headers(token):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_release(repo: str, tag: str, token):
    """Fetch a release by tag. Returns None if the tag has no release yet."""
    url = f"{GITHUB_API}/repos/{repo}/releases/tags/{tag}"
    try:
        response = requests.get(
            url, headers=_headers(token), timeout=REQUEST_TIMEOUT_SECONDS
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query release '{tag}' in '{repo}': {e}") from e
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API returned HTTP {response.status_code} fetching release "
            f"'{tag}' in '{repo}': {response.text}"
        )
    return response.json()


def fetch_assets(repo: str, release_id: int, token):
    """Fetch every asset on a release, following pagination to completion."""
    assets = []
    for page in range(1, MAX_ASSET_PAGES + 1):
        url = f"{GITHUB_API}/repos/{repo}/releases/{release_id}/assets"
        try:
            response = requests.get(
                url,
                headers=_headers(token),
                params={"per_page": ASSETS_PER_PAGE, "page": page},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to list assets for release {release_id} in '{repo}': {e}"
            ) from e
        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned HTTP {response.status_code} listing assets "
                f"for release {release_id} in '{repo}': {response.text}"
            )
        batch = response.json()
        assets.extend(batch)
        if len(batch) < ASSETS_PER_PAGE:
            return assets
    raise RuntimeError(
        f"Asset pagination for release {release_id} in '{repo}' exceeded "
        f"{MAX_ASSET_PAGES} pages; refusing to continue."
    )


def delete_asset(repo: str, asset_id: int, token):
    url = f"{GITHUB_API}/repos/{repo}/releases/assets/{asset_id}"
    try:
        response = requests.delete(
            url, headers=_headers(token), timeout=REQUEST_TIMEOUT_SECONDS
        )
    except Exception as e:
        raise RuntimeError(f"Failed to delete asset {asset_id} in '{repo}': {e}") from e
    # 204 on success; 404 means someone else already removed it, which is fine.
    if response.status_code not in (204, 404):
        raise RuntimeError(
            f"GitHub API returned HTTP {response.status_code} deleting asset "
            f"{asset_id} in '{repo}': {response.text}"
        )


def format_report(plan: PruningPlan, total_assets: int, retention_days: int) -> str:
    lines = []
    lines.append(
        f"Threshold: {plan.assets_per_run} assets/run x {retention_days} days "
        f"+ {plan.persistent_count} persistent = {plan.uncapped_threshold}"
        + (
            f", capped to {plan.threshold}"
            if plan.threshold < plan.uncapped_threshold
            else ""
        )
    )
    lines.append(f"Release assets before pruning: {total_assets}")
    if plan.doomed_versions:
        freed = sum(a.get("size", 0) for a in plan.assets_to_delete)
        lines.append(
            f"Pruning {len(plan.doomed_versions)} version group(s) "
            f"({len(plan.assets_to_delete)} assets, {freed / 1e9:.2f} GB):"
        )
        for version in plan.doomed_versions:
            lines.append(f"  - {version}")
    else:
        lines.append("Nothing to prune.")
    lines.append(
        f"Assets after pruning: {plan.kept_asset_count} "
        f"(+{plan.projected_asset_count - plan.kept_asset_count} incoming = "
        f"{plan.projected_asset_count} / {plan.threshold})"
    )
    for warning in plan.warnings:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Prune stale wheels from a rolling GitHub release."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Target repository as owner/name (defaults to $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--tag", default="dev-wheels", help="Release tag to prune")
    parser.add_argument("--package", default="torch-mlir", help="Package name")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help="Days of history to retain; converted to an asset count via the rate",
    )
    parser.add_argument(
        "--incoming",
        type=int,
        default=0,
        help="Number of wheels the current run is about to upload",
    )
    parser.add_argument(
        "--assets-per-run",
        type=int,
        default=None,
        help=(
            "Wheels published per run. Defaults to --incoming, falling back to "
            "the largest version group already on the release."
        ),
    )
    parser.add_argument(
        "--hard-limit",
        type=int,
        default=DEFAULT_HARD_LIMIT,
        help=(
            f"Cap on the computed threshold, so a wider matrix cannot exceed "
            f"GitHub's {GITHUB_MAX_RELEASE_ASSETS}-asset limit"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the plan without deleting anything",
    )
    args = parser.parse_args()

    if not args.repo:
        parser.error("--repo is required when $GITHUB_REPOSITORY is unset")
    if args.hard_limit > GITHUB_MAX_RELEASE_ASSETS:
        parser.error(
            f"--hard-limit {args.hard_limit} exceeds GitHub's limit of "
            f"{GITHUB_MAX_RELEASE_ASSETS}"
        )

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    release = fetch_release(args.repo, args.tag, token)
    if release is None:
        print(
            f"Release '{args.tag}' does not exist in '{args.repo}'; nothing to prune."
        )
        return

    assets = fetch_assets(args.repo, release["id"], token)
    plan = plan_pruning(
        assets,
        package_name=args.package,
        retention_days=args.retention_days,
        incoming=args.incoming,
        assets_per_run=args.assets_per_run,
        hard_limit=args.hard_limit,
    )

    report = format_report(plan, len(assets), args.retention_days)
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"### Dev-wheels asset pruning\n\n```\n{report}\n```\n")

    if args.dry_run:
        print("Dry run: no assets deleted.")
        return

    for asset in plan.assets_to_delete:
        print(f"Deleting {asset['name']}")
        delete_asset(args.repo, asset["id"], token)

    if plan.assets_to_delete:
        print(f"Deleted {len(plan.assets_to_delete)} asset(s).")


if __name__ == "__main__":
    sys.exit(main())
