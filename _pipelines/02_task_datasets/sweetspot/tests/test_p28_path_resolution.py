"""Regression tests for P28/P29 root resolution.

Both modules used to derive their path constants by counting parent directories
(``HERE.parents[2]``, ``WORKTREE_ROOT.parents[2]``). That only holds under a
``.claude/worktrees/<name>/`` layout: on a mainline checkout ``PROJECT_ROOT``
resolved to ``/mnt/data`` and ``REFERENCE_ROOT`` to a nonexistent sibling, so
``_reference_inputs()`` died in ``git rev-parse HEAD``.

These tests pin the replacement behaviour: roots come from git, and the
reference root degrades to the main repository instead of a hard-coded sibling.
"""
from __future__ import annotations

import ast
import importlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


p28 = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p28_agentic_optimization"
)
p29 = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p29_agent_action_effect"
)


class SweetspotPathResolutionTests(unittest.TestCase):
    def test_checkout_root_matches_git_toplevel(self) -> None:
        expected = subprocess.check_output(
            ["git", "-C", str(Path(p28.__file__).parent), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
        self.assertEqual(p28.WORKTREE_ROOT, Path(expected).resolve())

    def test_project_root_is_main_repo_not_a_parent_count(self) -> None:
        common = subprocess.check_output(
            ["git", "-C", str(Path(p28.__file__).parent), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
        self.assertEqual(p28.PROJECT_ROOT, Path(common).resolve().parent)

    def test_project_root_carries_the_shared_volve_data(self) -> None:
        """PROJECT_ROOT must point at the checkout that owns the shared dataset.

        The old `.parents[2]` form resolved to `/mnt/data` on a mainline
        checkout, which exists and therefore failed silently rather than loudly.
        """
        self.assertTrue(
            (p28.PROJECT_ROOT / "_sandbox/volve_data").is_dir(),
            f"shared Volve data missing under resolved PROJECT_ROOT {p28.PROJECT_ROOT}",
        )

    def test_p29_reuses_p28_roots(self) -> None:
        self.assertEqual(p29.WORKTREE_ROOT, p28.WORKTREE_ROOT)
        self.assertEqual(p29.PROJECT_ROOT, p28.PROJECT_ROOT)
        self.assertEqual(p29.REFERENCE_ROOT, p28.REFERENCE_ROOT)

    def test_reference_root_falls_back_to_main_repo_without_legacy_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "some-checkout"
            checkout.mkdir()
            main_repo = Path(tmp) / "main-repo"
            main_repo.mkdir()
            resolved = p28._resolve_reference_root(main_repo, checkout)
        self.assertEqual(resolved, main_repo)

    def test_reference_root_prefers_legacy_worktree_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "some-checkout"
            checkout.mkdir()
            main_repo = Path(tmp) / "main-repo"
            main_repo.mkdir()
            legacy = Path(tmp) / p28.REFERENCE_WORKTREE_NAME
            (legacy / "_pipelines/02_task_datasets/sweetspot").mkdir(parents=True)
            resolved = p28._resolve_reference_root(main_repo, checkout)
        self.assertEqual(resolved, legacy.resolve())

    def test_reference_root_env_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "some-checkout"
            checkout.mkdir()
            main_repo = Path(tmp) / "main-repo"
            main_repo.mkdir()
            legacy = Path(tmp) / p28.REFERENCE_WORKTREE_NAME
            (legacy / "_pipelines/02_task_datasets/sweetspot").mkdir(parents=True)
            override = Path(tmp) / "explicit-reference"
            override.mkdir()
            previous = os.environ.get(p28.REFERENCE_ROOT_ENV)
            os.environ[p28.REFERENCE_ROOT_ENV] = str(override)
            try:
                resolved = p28._resolve_reference_root(main_repo, checkout)
            finally:
                if previous is None:
                    del os.environ[p28.REFERENCE_ROOT_ENV]
                else:
                    os.environ[p28.REFERENCE_ROOT_ENV] = previous
        self.assertEqual(resolved, override.resolve())

    def test_main_repo_can_serve_every_reference_artifact(self) -> None:
        """The fallback is only safe if the main repo actually holds the inputs.

        Each of these is version-controlled and byte-identical to the legacy
        worktree's copy; this test fails loudly if that stops being true.
        """
        required = [
            p28.P5_LABEL_MAPPING_ID,
            p28.P5_T3_SPLIT_MANIFEST_ID,
            p28.P5_STAGE3_T3_LEADERBOARD_ID,
            p28.P5_STAGE3_SUMMARY_ID,
            p28.P5_STAGE4_SUMMARY_ID,
            p28.P7_SUMMARY_ID,
            p28.P8_SUMMARY_ID,
        ]
        missing = [rel for rel in required if not (p28.PROJECT_ROOT / rel).is_file()]
        self.assertEqual(missing, [], f"main repo cannot serve reference inputs: {missing}")

    def test_no_module_derives_roots_by_parent_counting(self) -> None:
        """Guard against a regression to `parents[N]` root derivation.

        `p29` keeps one `parents[3]` for its pre-import sys.path bootstrap, which
        is layout-independent; everything else must go through git.
        """
        offenders: list[str] = []
        for module, allowed in ((p28, 0), (p29, 1)):
            source = Path(module.__file__).read_text(encoding="utf-8")
            hits = 0
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Subscript):
                    continue
                value = node.value
                if isinstance(value, ast.Attribute) and value.attr == "parents":
                    hits += 1
            if hits > allowed:
                offenders.append(f"{Path(module.__file__).name}: {hits} > {allowed}")
        self.assertEqual(offenders, [], f"root derivation by parent counting reappeared: {offenders}")


if __name__ == "__main__":
    unittest.main()
