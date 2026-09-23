# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

"""End-to-end simulation of the PyPI partial-upload retry runbook.

`.github/workflows/release.yml` documents what an operator must do when the
`pypi_upload` job dies partway through a release: click "Re-run failed jobs"
rather than starting a new run. These tests execute that procedure against a
fake PyPI implementation.

The simulation models the two facts that make the retry path different from the
first attempt:

  * "Re-run failed jobs" re-runs only the failed job. `prepare_pypi` already
    succeeded, so it does not run again and its `pypi-ready-dist` artifact is
    *not* re-pruned: the retry re-offers every wheel that was missing before the
    FIRST attempt, including the ones that attempt successfully published.
  * PyPI rejects an upload whose filename already exists, whatever its content.

The configuration the simulation depends on (whether the publisher skips
existing files, which artifact moves between the jobs) is read out of the real
workflow file, so these tests fail if the workflow stops implementing the
documented procedure.
"""

import hashlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is in build-requirements.txt
    yaml = None

# Add repo root to sys.path so scripts module can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import prepare_pypi_publish

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"

PACKAGE = "torch-mlir"
VERSION = "20260901"
PLATFORMS = [
    "linux_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_11_0_arm64",
    "win_amd64",
]


def wheel_name(platform: str) -> str:
    return f"torch_mlir-{VERSION}-cp311-cp311-{platform}.whl"


def build_wheels(salt: bytes = b"") -> dict[str, bytes]:
    """Produce the wheel set a `build_wheels` job would upload as `dist-*`.

    `salt` models a rebuild: recompiling produces new zip timestamps, so the
    bytes (and therefore the SHA256) of an otherwise identical wheel change.
    """
    return {wheel_name(p): b"wheel:" + p.encode() + salt for p in PLATFORMS}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().lower()


class PublishError(RuntimeError):
    """A failure of the publish step, e.g. a dropped connection to PyPI."""


class FakePyPI:
    """Minimal model of PyPI upload + JSON API semantics.

    Uploads are rejected by *filename*: PyPI never lets a filename be reused,
    and does not care whether the content matches what it already has. That
    indifference is why `skip-existing` alone cannot tell "this is the wheel my
    own failed attempt uploaded" from "this is somebody else's file".
    """

    def __init__(self):
        self.files: dict[str, str] = {}  # filename -> sha256
        self.accepted_uploads: list[str] = []

    def upload(self, filename: str, content: bytes) -> None:
        if filename in self.files:
            raise FileExistsError(
                f"400 Bad Request: File already exists ('{filename}')."
            )
        self.files[filename] = sha256_bytes(content)
        self.accepted_uploads.append(filename)

    def json_api(self, package_name: str, version_str: str) -> dict[str, str]:
        """Stand-in for prepare_pypi_publish.fetch_pypi_release_files."""
        assert package_name == PACKAGE and version_str == VERSION
        return dict(self.files)


class FakePublisher:
    """Models `twine upload [--skip-existing]`."""

    def __init__(
        self, pypi: FakePyPI, skip_existing: bool, fail_after_uploads: int | None = None
    ):
        self.pypi = pypi
        self.skip_existing = skip_existing
        self.fail_after_uploads = fail_after_uploads
        self.uploaded: list[str] = []
        self.skipped: list[str] = []

    def publish(self, dist_dir: pathlib.Path) -> None:
        for path in sorted(dist_dir.glob("*.whl")):
            if (
                self.fail_after_uploads is not None
                and len(self.uploaded) >= self.fail_after_uploads
            ):
                raise PublishError(
                    f"simulated upload failure before '{path.name}' "
                    f"(uploaded {self.uploaded})"
                )
            try:
                self.pypi.upload(path.name, path.read_bytes())
            except FileExistsError:
                if not self.skip_existing:
                    raise
                self.skipped.append(path.name)
                continue
            self.uploaded.append(path.name)


class WorkflowConfig:
    """The parts of release.yml that the simulation reproduces."""

    def __init__(self, workflow: dict):
        jobs = workflow["jobs"]
        self.prepare_job = jobs["prepare_pypi"]
        self.upload_job = jobs["pypi_upload"]

        self.publish_step = self._step(
            self.upload_job,
            lambda s: "pypa/gh-action-pypi-publish" in s.get("uses", ""),
        )
        self.skip_existing = self._as_bool(
            self.publish_step.get("with", {}).get("skip-existing", False)
        )
        download_step = self._step(
            self.upload_job,
            lambda s: "actions/download-artifact" in str(s.get("uses", "")),
        )
        self.ready_artifact = download_step["with"]["name"]
        self.uploaded_artifacts = {
            step["with"]["name"]
            for step in self.prepare_job["steps"]
            if "actions/upload-artifact" in str(step.get("uses", ""))
        }

    @staticmethod
    def _as_bool(value) -> bool:
        return str(value).lower() == "true"

    @staticmethod
    def _step(job: dict, predicate):
        for step in job["steps"]:
            if predicate(step):
                return step
        raise AssertionError(f"no matching step in job '{job.get('name')}'")


def load_workflow_config() -> WorkflowConfig:
    return WorkflowConfig(yaml.safe_load(RELEASE_WORKFLOW.read_text()))


class SimulatedRun:
    """One GitHub Actions run of the PyPI half of the release workflow.

    Artifacts are snapshots: a job uploads a copy, and every download gets a
    fresh copy of that same snapshot. That is the crux of the bug these tests
    cover, since re-running `pypi_upload` alone downloads the snapshot taken
    before the first upload attempt.
    """

    def __init__(
        self,
        pypi: FakePyPI,
        config: WorkflowConfig,
        root: pathlib.Path,
        wheels: dict[str, bytes],
    ):
        self.pypi = pypi
        self.config = config
        self.root = root
        self.build_artifact = wheels
        self.artifacts: dict[str, dict[str, bytes]] = {}
        self.job_counter = 0
        self.prepare_runs = 0

    def _workspace(self, job_name: str) -> pathlib.Path:
        self.job_counter += 1
        path = self.root / f"{self.job_counter:02d}-{job_name}"
        path.mkdir(parents=True)
        return path

    def _materialize(
        self, workspace: pathlib.Path, files: dict[str, bytes]
    ) -> pathlib.Path:
        dist = workspace / "dist"
        dist.mkdir()
        for name, content in files.items():
            (dist / name).write_bytes(content)
        return dist

    def run_prepare_pypi(self) -> tuple[bool, str]:
        """Job `prepare_pypi`: validate against PyPI, prune, publish artifact."""
        self.prepare_runs += 1
        workspace = self._workspace("prepare_pypi")
        dist = self._materialize(workspace, self.build_artifact)

        with mock.patch(
            "scripts.prepare_pypi_publish.fetch_pypi_release_files",
            side_effect=self.pypi.json_api,
        ):
            should_upload, state = prepare_pypi_publish.prepare_publish(
                dist, PACKAGE, VERSION
            )

        if should_upload:
            self.artifacts[self.config.ready_artifact] = {
                path.name: path.read_bytes() for path in sorted(dist.glob("*.whl"))
            }
        return should_upload, state

    def run_pypi_upload(self, publisher: FakePublisher) -> None:
        """Job `pypi_upload`: download the artifact, hand it to the publisher."""
        workspace = self._workspace("pypi_upload")
        dist = self._materialize(workspace, self.artifacts[self.config.ready_artifact])
        publisher.publish(dist)


@unittest.skipIf(yaml is None, "pyyaml is required to read the release workflow")
class TestReleaseWorkflowContract(unittest.TestCase):
    """Pin the workflow configuration the runbook (and simulation) relies on."""

    def setUp(self):
        self.config = load_workflow_config()

    def test_publisher_skips_already_published_files(self):
        self.assertTrue(
            self.config.skip_existing,
            "pypi_upload must set skip-existing: true, otherwise 'Re-run failed "
            "jobs' dies on the wheels the failed attempt already published.",
        )

    def test_oidc_job_contains_no_project_code(self):
        """The publishing job holds the OIDC token; keep it free of our code."""
        self.assertEqual(self.config.upload_job["permissions"], {"id-token": "write"})
        for step in self.config.upload_job["steps"]:
            self.assertNotIn(
                "run",
                step,
                "the OIDC job must not run project scripts; validation belongs "
                "in prepare_pypi",
            )
            self.assertNotIn("actions/checkout", str(step.get("uses", "")))

    def test_upload_job_consumes_the_artifact_prepare_produces(self):
        self.assertIn(self.config.ready_artifact, self.config.uploaded_artifacts)


@unittest.skipIf(yaml is None, "pyyaml is required to read the release workflow")
class TestPartialUploadRetry(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.config = load_workflow_config()
        self.pypi = FakePyPI()
        self.wheels = build_wheels()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _new_run(self, wheels: dict[str, bytes] | None = None) -> SimulatedRun:
        root = self.root / f"run{len(list(self.root.iterdir())) + 1}"
        root.mkdir()
        return SimulatedRun(self.pypi, self.config, root, wheels or self.wheels)

    def _partially_failed_first_attempt(self) -> SimulatedRun:
        """First attempt: prepare succeeds, publishing dies after two wheels."""
        run = self._new_run()
        should_upload, state = run.run_prepare_pypi()
        self.assertTrue(should_upload)
        self.assertEqual(state, "FRESH_RELEASE")

        publisher = FakePublisher(
            self.pypi, skip_existing=self.config.skip_existing, fail_after_uploads=2
        )
        with self.assertRaises(PublishError):
            run.run_pypi_upload(publisher)

        self.assertEqual(len(self.pypi.files), 2)
        self.assertEqual(len(publisher.uploaded), 2)
        return run

    def test_runbook_rerun_failed_jobs_completes_the_release(self):
        """The documented procedure, executed against a partial release."""
        run = self._partially_failed_first_attempt()
        published_by_first_attempt = set(self.pypi.files)

        # "Re-run failed jobs": prepare_pypi is NOT re-run, so pypi_upload
        # re-downloads the unpruned artifact from before the first attempt.
        retry_publisher = FakePublisher(
            self.pypi, skip_existing=self.config.skip_existing
        )
        run.run_pypi_upload(retry_publisher)

        self.assertEqual(
            run.prepare_runs, 1, "prepare_pypi must not need to re-run on retry"
        )
        self.assertEqual(set(retry_publisher.skipped), published_by_first_attempt)
        self.assertEqual(
            set(retry_publisher.uploaded),
            set(self.wheels) - published_by_first_attempt,
        )
        # Every wheel published exactly once, with the bytes that were built.
        self.assertEqual(sorted(self.pypi.accepted_uploads), sorted(self.wheels))
        self.assertEqual(
            self.pypi.files,
            {name: sha256_bytes(content) for name, content in self.wheels.items()},
        )

    def test_retry_without_skip_existing_is_the_reported_bug(self):
        """Regression guard: this is what happens if skip-existing is dropped."""
        run = self._partially_failed_first_attempt()
        stuck_at = sorted(self.pypi.files)[0]

        retry_publisher = FakePublisher(self.pypi, skip_existing=False)
        with self.assertRaises(FileExistsError) as ctx:
            run.run_pypi_upload(retry_publisher)
        self.assertIn(stuck_at, str(ctx.exception))

        # The release is stuck: two wheels are on PyPI and no amount of
        # re-running this job can deliver the rest.
        self.assertEqual(len(self.pypi.files), 2)
        self.assertEqual(retry_publisher.uploaded, [])

    def test_new_workflow_run_with_rebuilt_wheels_fails_closed(self):
        """Runbook step 1: do NOT dispatch a fresh run for the same version."""
        self._partially_failed_first_attempt()

        rebuilt = self._new_run(wheels=build_wheels(salt=b"-rebuilt"))
        with self.assertRaises(ValueError) as ctx:
            rebuilt.run_prepare_pypi()
        self.assertIn("Content collision", str(ctx.exception))
        self.assertIn("PyPI artifacts are immutable", str(ctx.exception))

    def test_rerunning_prepare_with_identical_wheels_prunes_and_skips_upload(self):
        """A re-run of every job is safe if the wheels are byte-identical."""
        run = self._partially_failed_first_attempt()
        run.run_pypi_upload(
            FakePublisher(self.pypi, skip_existing=self.config.skip_existing)
        )

        rerun = self._new_run()
        should_upload, state = rerun.run_prepare_pypi()
        self.assertFalse(should_upload)
        self.assertEqual(state, "ALREADY_COMPLETED")
        # pypi_upload is skipped by its `if:` condition.
        self.assertNotIn(self.config.ready_artifact, rerun.artifacts)

    def test_preflight_is_the_guard_that_skip_existing_relies_on(self):
        """With skip-existing on, prepare_pypi is what enforces immutability.

        A filename already served by PyPI with foreign content never reaches the
        publisher: the preflight fails closed before the artifact is built.
        """
        hijacked = wheel_name(PLATFORMS[0])
        self.pypi.upload(hijacked, b"content published by something else")

        run = self._new_run()
        with self.assertRaises(ValueError) as ctx:
            run.run_prepare_pypi()
        self.assertIn("Content collision", str(ctx.exception))
        self.assertIn(hijacked, str(ctx.exception))
        self.assertNotIn(self.config.ready_artifact, run.artifacts)

    def test_known_gap_foreign_file_landing_after_the_preflight_is_skipped(self):
        """Documents the accepted residual risk of `skip-existing: true`.

        If a filename appears on PyPI with different content *after* the
        preflight ran, the publisher skips it silently instead of failing. This
        needs an upload from outside this workflow (a maintainer running twine
        by hand, say) during the window between the two jobs; a rebuilt release
        run cannot cause it, because that run's own preflight fails closed --
        see test_new_workflow_run_with_rebuilt_wheels_fails_closed.
        """
        run = self._new_run()
        run.run_prepare_pypi()

        hijacked = sorted(self.wheels)[0]
        self.pypi.upload(hijacked, b"content published by something else")

        publisher = FakePublisher(self.pypi, skip_existing=self.config.skip_existing)
        run.run_pypi_upload(publisher)

        self.assertEqual(publisher.skipped, [hijacked])
        self.assertNotEqual(
            self.pypi.files[hijacked], sha256_bytes(self.wheels[hijacked])
        )


if __name__ == "__main__":
    unittest.main()
