"""Tests for scripts/prune_release_assets.py."""

import datetime
import unittest

import prune_release_assets as prune

NOW = datetime.datetime(2026, 9, 23, 12, 0, 0, tzinfo=datetime.timezone.utc)

PLATFORMS = [
    "macosx_11_0_arm64",
    "manylinux_2_27_aarch64.manylinux_2_28_aarch64",
    "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
    "win_amd64",
]
PY_TAGS = [("cp310", "cp310"), ("cp311", "cp311"), ("cp312", "abi3")]

# The current matrix: 4 platforms x 3 CPython builds.
WHEELS_PER_RUN = len(PLATFORMS) * len(PY_TAGS)

BINARY_NAMES = [
    "torch-mlir-opt-macosx_11_0_arm64",
    "torch-mlir-opt-manylinux_2_27_aarch64",
    "torch-mlir-opt-manylinux_2_27_x86_64",
    "torch-mlir-opt-win_amd64.exe",
]


def make_asset(name, created_at, asset_id=None, size=1000):
    return {
        "id": asset_id if asset_id is not None else abs(hash(name)) % 10**9,
        "name": name,
        "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "size": size,
    }


def make_wheel_group(version, created_at, platforms=None, py_tags=None):
    """Build the wheels a single run produces for one version."""
    assets = []
    for python, abi in py_tags or PY_TAGS:
        for platform in platforms or PLATFORMS:
            name = f"torch_mlir-{version}-{python}-{abi}-{platform}.whl"
            assets.append(make_asset(name, created_at))
    return assets


def make_history(num_days, end=NOW, py_tags=None):
    """Build `num_days` of daily dev releases ending the day before `end`."""
    assets = []
    for days_ago in range(1, num_days + 1):
        created = end - datetime.timedelta(days=days_ago)
        version = f"{created.strftime('%Y%m%d')}.dev0"
        assets.extend(make_wheel_group(version, created, py_tags=py_tags))
    return assets


def make_binaries(created_at=NOW):
    return [make_asset(name, created_at, size=50_000_000) for name in BINARY_NAMES]


def version_days_ago(days):
    return (NOW - datetime.timedelta(days=days)).strftime("%Y%m%d") + ".dev0"


class ParseWheelVersionTest(unittest.TestCase):
    def setUp(self):
        self.escaped = prune.escape_package_name("torch-mlir")

    def test_parses_standard_wheel(self):
        name = "torch_mlir-20260922.dev0-cp310-cp310-win_amd64.whl"
        self.assertEqual(prune.parse_wheel_version(name, self.escaped), "20260922.dev0")

    def test_parses_abi3_wheel(self):
        name = (
            "torch_mlir-20260923.dev0-cp312-abi3-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        )
        self.assertEqual(prune.parse_wheel_version(name, self.escaped), "20260923.dev0")

    def test_parses_non_dev_version(self):
        name = "torch_mlir-20261001-cp311-cp311-macosx_11_0_arm64.whl"
        self.assertEqual(prune.parse_wheel_version(name, self.escaped), "20261001")

    def test_rejects_unversioned_binaries(self):
        for name in BINARY_NAMES:
            self.assertIsNone(prune.parse_wheel_version(name, self.escaped), name)

    def test_rejects_other_packages(self):
        name = "some_other_pkg-1.2.3-cp310-cp310-win_amd64.whl"
        self.assertIsNone(prune.parse_wheel_version(name, self.escaped))

    def test_rejects_non_wheel_extensions(self):
        self.assertIsNone(
            prune.parse_wheel_version("torch_mlir-20260922.dev0.tar.gz", self.escaped)
        )

    def test_unparseable_names_are_treated_as_unprunable(self):
        # PEP 427 requires the escaped distribution name ('torch_mlir'), so a
        # hyphenated spelling is not a legal wheel name. Anything we cannot
        # confidently attribute to a version is classified persistent and left
        # alone, since the cost of a false positive is deleting a live wheel.
        name = "torch-mlir-20260922.dev0-cp310-cp310-win_amd64.whl"
        self.assertIsNone(prune.parse_wheel_version(name, self.escaped))


class ParseTimestampTest(unittest.TestCase):
    def test_parses_github_format(self):
        parsed = prune.parse_timestamp("2026-09-23T13:19:44Z")
        self.assertEqual(
            parsed,
            datetime.datetime(2026, 9, 23, 13, 19, 44, tzinfo=datetime.timezone.utc),
        )

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            prune.parse_timestamp("not-a-timestamp")


class GroupAssetsTest(unittest.TestCase):
    def test_splits_wheels_from_persistent_assets(self):
        assets = make_wheel_group("20260922.dev0", NOW) + make_binaries()
        groups, persistent = prune.group_assets(assets, "torch-mlir")
        self.assertEqual(list(groups), ["20260922.dev0"])
        self.assertEqual(len(groups["20260922.dev0"].assets), WHEELS_PER_RUN)
        self.assertEqual(len(persistent), 4)

    def test_tolerates_missing_created_at(self):
        group = prune.VersionGroup("20260922.dev0")
        group.add({"id": 1, "name": "x.whl", "size": 1})
        self.assertEqual(group.newest_created_at, prune._EPOCH)


class ThresholdComputationTest(unittest.TestCase):
    """The threshold is derived from the rate, not hardcoded."""

    def test_current_matrix_sixty_days(self):
        # 12 wheels/run x 60 days + 4 persistent binaries.
        threshold, uncapped = prune.compute_threshold(12, 60, 4, 990)
        self.assertEqual(uncapped, 724)
        self.assertEqual(threshold, 724)

    def test_threshold_tracks_a_wider_matrix(self):
        # Adding cp313 takes the matrix to 16 wheels/run.
        threshold, uncapped = prune.compute_threshold(16, 60, 4, 990)
        self.assertEqual(uncapped, 964)
        self.assertEqual(threshold, 964)

    def test_threshold_is_capped_at_the_hard_limit(self):
        # cp313 + cp314 would need 1,204 assets, above GitHub's limit.
        threshold, uncapped = prune.compute_threshold(20, 60, 4, 990)
        self.assertEqual(uncapped, 1204)
        self.assertEqual(threshold, 990)

    def test_threshold_scales_with_retention_days(self):
        self.assertEqual(prune.compute_threshold(12, 30, 4, 990)[0], 364)
        self.assertEqual(prune.compute_threshold(12, 1, 4, 990)[0], 16)


class InferRateTest(unittest.TestCase):
    def test_prefers_the_current_runs_upload_count(self):
        groups, _ = prune.group_assets(make_history(3), "torch-mlir")
        self.assertEqual(prune.infer_assets_per_run(groups, incoming=16), 16)

    def test_falls_back_to_largest_existing_group(self):
        groups, _ = prune.group_assets(make_history(3), "torch-mlir")
        self.assertEqual(prune.infer_assets_per_run(groups, incoming=0), WHEELS_PER_RUN)

    def test_ignores_partial_groups_when_inferring(self):
        # A run that failed on Windows left a short group; the rate should come
        # from the complete one, not the truncated one.
        assets = make_wheel_group("20260922.dev0", NOW, platforms=PLATFORMS[:2])
        assets += make_wheel_group("20260923.dev0", NOW)
        groups, _ = prune.group_assets(assets, "torch-mlir")
        self.assertEqual(prune.infer_assets_per_run(groups, incoming=0), WHEELS_PER_RUN)

    def test_empty_release(self):
        self.assertEqual(prune.infer_assets_per_run({}, incoming=0), 0)


class RetentionWindowTest(unittest.TestCase):
    """The count threshold should preserve exactly `retention_days` of builds."""

    def test_steady_state_keeps_exactly_sixty_days(self):
        # 59 days of history already present, plus today's 12 incoming = 60.
        assets = make_history(59) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.threshold, 724)
        self.assertEqual(plan.doomed_versions, [])
        self.assertEqual(plan.projected_asset_count, 724)

    def test_one_day_over_prunes_exactly_one_group(self):
        assets = make_history(60) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.doomed_versions, [version_days_ago(60)])
        self.assertEqual(len(plan.assets_to_delete), WHEELS_PER_RUN)
        self.assertEqual(plan.projected_asset_count, 724)

    def test_large_backlog_is_trimmed_to_the_window(self):
        assets = make_history(200) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(len(plan.doomed_versions), 141)
        # 59 retained days + 1 incoming = 60 days of builds.
        self.assertEqual(plan.kept_asset_count, 59 * WHEELS_PER_RUN + 4)
        self.assertEqual(plan.projected_asset_count, 724)

    def test_prunes_oldest_first(self):
        assets = make_history(200) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.doomed_versions[0], version_days_ago(200))
        self.assertEqual(plan.doomed_versions[-1], version_days_ago(60))
        self.assertNotIn(version_days_ago(59), plan.doomed_versions)
        self.assertNotIn(version_days_ago(1), plan.doomed_versions)

    def test_under_the_window_prunes_nothing(self):
        assets = make_history(10) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.doomed_versions, [])


class OutageResilienceTest(unittest.TestCase):
    """A count policy must not erode the archive when nothing is published.

    This is the key advantage over an age-based policy, which would age the
    whole archive out during a prolonged outage.
    """

    def test_stale_archive_is_untouched_when_nothing_is_incoming(self):
        # Every build is a year old and the nightly has not run since.
        assets = make_history(60, end=NOW - datetime.timedelta(days=365))
        assets += make_binaries(NOW - datetime.timedelta(days=365))
        plan = prune.plan_pruning(assets, "torch-mlir", retention_days=60, incoming=0)
        self.assertEqual(plan.doomed_versions, [])
        self.assertEqual(plan.assets_to_delete, [])

    def test_resumed_nightly_after_outage_keeps_full_window(self):
        assets = make_history(59, end=NOW - datetime.timedelta(days=100))
        assets += make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.doomed_versions, [])


class GroupAtomicityTest(unittest.TestCase):
    def test_never_leaves_a_partial_version(self):
        assets = make_history(200) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        deleted_names = set(a["name"] for a in plan.assets_to_delete)
        groups, _ = prune.group_assets(assets, "torch-mlir")
        for version, group in groups.items():
            names = set(a["name"] for a in group.assets)
            overlap = names & deleted_names
            # Each version group is either fully deleted or fully retained.
            self.assertIn(
                overlap, (set(), names), f"version {version} was partially pruned"
            )

    def test_deletes_incomplete_groups_wholly(self):
        # A run that failed on Windows left only 9 wheels for the oldest version.
        assets = make_wheel_group(version_days_ago(200), NOW, platforms=PLATFORMS[:3])
        assets += make_history(60)
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.doomed_versions[0], version_days_ago(200))
        # All 9 of the partial group's assets go, not just enough to fit.
        doomed_assets = [
            a for a in plan.assets_to_delete if version_days_ago(200) in a["name"]
        ]
        self.assertEqual(len(doomed_assets), 9)


class PersistentAssetTest(unittest.TestCase):
    def test_binaries_are_never_pruned(self):
        stale_binaries = make_binaries(NOW - datetime.timedelta(days=365))
        assets = make_history(200) + stale_binaries
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        deleted_names = set(a["name"] for a in plan.assets_to_delete)
        for name in BINARY_NAMES:
            self.assertNotIn(name, deleted_names)

    def test_persistent_assets_are_budgeted_into_the_threshold(self):
        with_binaries = prune.plan_pruning(
            make_history(100) + make_binaries(),
            "torch-mlir",
            retention_days=60,
            incoming=WHEELS_PER_RUN,
        )
        without_binaries = prune.plan_pruning(
            make_history(100),
            "torch-mlir",
            retention_days=60,
            incoming=WHEELS_PER_RUN,
        )
        # The 4 binaries raise the threshold by exactly 4, so the same number of
        # wheel groups survives either way.
        self.assertEqual(with_binaries.threshold, 724)
        self.assertEqual(without_binaries.threshold, 720)
        self.assertEqual(
            len(with_binaries.doomed_versions), len(without_binaries.doomed_versions)
        )


class HardLimitTest(unittest.TestCase):
    def test_wider_matrix_is_capped_and_warns(self):
        # 5 CPython builds x 4 platforms = 20 wheels/run.
        py_tags = PY_TAGS + [("cp313", "cp313"), ("cp314", "cp314")]
        assets = make_history(60, py_tags=py_tags) + make_binaries()
        plan = prune.plan_pruning(assets, "torch-mlir", retention_days=60, incoming=20)
        self.assertEqual(plan.uncapped_threshold, 1204)
        self.assertEqual(plan.threshold, 990)
        self.assertLessEqual(plan.projected_asset_count, 990)
        self.assertLess(plan.projected_asset_count, prune.GITHUB_MAX_RELEASE_ASSETS)
        self.assertEqual(plan.effective_retention_days, 49)
        self.assertTrue(any("Capping" in w for w in plan.warnings), plan.warnings)

    def test_no_warning_when_the_window_fits(self):
        assets = make_history(60) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.warnings, [])
        self.assertEqual(plan.effective_retention_days, 60)

    def test_default_hard_limit_is_below_githubs_limit(self):
        self.assertLess(prune.DEFAULT_HARD_LIMIT, prune.GITHUB_MAX_RELEASE_ASSETS)

    def test_never_deletes_the_last_version(self):
        assets = make_history(3)
        plan = prune.plan_pruning(
            assets,
            "torch-mlir",
            retention_days=1,
            incoming=WHEELS_PER_RUN,
            assets_per_run=1,
        )
        self.assertEqual(len(plan.doomed_versions), 2)
        self.assertTrue(any("Cannot fit" in w for w in plan.warnings), plan.warnings)


class OverrideTest(unittest.TestCase):
    def test_explicit_assets_per_run_overrides_inference(self):
        assets = make_history(100) + make_binaries()
        plan = prune.plan_pruning(
            assets,
            "torch-mlir",
            retention_days=60,
            incoming=WHEELS_PER_RUN,
            assets_per_run=6,
        )
        self.assertEqual(plan.threshold, 6 * 60 + 4)

    def test_dry_run_without_incoming_infers_the_rate(self):
        assets = make_history(100) + make_binaries()
        plan = prune.plan_pruning(assets, "torch-mlir", retention_days=60, incoming=0)
        self.assertEqual(plan.assets_per_run, WHEELS_PER_RUN)
        self.assertEqual(plan.threshold, 724)


class EdgeCaseTest(unittest.TestCase):
    def test_empty_release(self):
        plan = prune.plan_pruning([], "torch-mlir", retention_days=60)
        self.assertEqual(plan.doomed_versions, [])
        self.assertEqual(plan.kept_asset_count, 0)

    def test_only_persistent_assets(self):
        plan = prune.plan_pruning(
            make_binaries(), "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.doomed_versions, [])
        self.assertEqual(plan.kept_asset_count, 4)

    def test_unparseable_version_sorts_oldest(self):
        assets = make_wheel_group("not!a!version", NOW - datetime.timedelta(days=90))
        assets += make_history(60)
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        self.assertEqual(plan.doomed_versions[0], "not!a!version")

    def test_multiple_runs_per_day_are_separate_groups(self):
        # A same-day re-run produces .dev1; both are independent groups, and the
        # count policy trims by group, so two runs a day halves the day window.
        created = NOW - datetime.timedelta(days=1)
        assets = make_wheel_group("20260922.dev0", created)
        assets += make_wheel_group("20260922.dev1", created)
        groups, _ = prune.group_assets(assets, "torch-mlir")
        self.assertEqual(len(groups), 2)

    def test_rejects_invalid_arguments(self):
        for kwargs in (
            {"retention_days": 0},
            {"incoming": -1},
            {"hard_limit": 0},
            {"assets_per_run": -1},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                prune.plan_pruning([], "torch-mlir", **kwargs)


class ReportTest(unittest.TestCase):
    def test_report_shows_the_threshold_derivation(self):
        assets = make_history(100) + make_binaries()
        plan = prune.plan_pruning(
            assets, "torch-mlir", retention_days=60, incoming=WHEELS_PER_RUN
        )
        report = prune.format_report(plan, len(assets), 60)
        self.assertIn("12 assets/run x 60 days + 4 persistent = 724", report)

    def test_report_shows_the_cap(self):
        assets = make_history(60, py_tags=PY_TAGS + [("cp313", "cp313")])
        assets += make_binaries()
        plan = prune.plan_pruning(assets, "torch-mlir", retention_days=80, incoming=16)
        report = prune.format_report(plan, len(assets), 80)
        self.assertIn("capped to 990", report)


if __name__ == "__main__":
    unittest.main()
