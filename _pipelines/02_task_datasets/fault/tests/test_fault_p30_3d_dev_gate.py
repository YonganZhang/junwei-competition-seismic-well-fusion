from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

TRACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK_DIR))

from fault_p30_3d_dev_gate import (  # noqa: E402
    PROJECT_ROOT,
    _build_dev_asset,
    DEV_BOX,
    DEV_SPLIT,
)


class FaultP30ThreeDDevGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_root = TRACK_DIR / "_outputs" / "p30_3d_dev_gate_test"
        if cls.output_root.exists():
            shutil.rmtree(cls.output_root)
        cls.result = _build_dev_asset(cls.output_root)
        cls.manifest_path = cls.output_root / "manifest.json"
        cls.split_manifest_path = cls.output_root / "split_manifest.json"
        cls.gate_result_path = cls.output_root / "gate_result.json"
        cls.subvolume_path = cls.output_root / "dev_subvolume.npz"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.output_root.exists():
            shutil.rmtree(cls.output_root)

    def test_gate_ready_and_not_holdout_accessed(self) -> None:
        gate = self.result["gate_result"]
        self.assertEqual(gate["status"], "READY")
        self.assertEqual(gate["reason_codes"], [])
        self.assertFalse(gate["frozen_holdout_accessed"])
        self.assertFalse(self.result["manifest"]["data_gate_blocked"])

    def test_subvolume_coordinates_and_masks_are_consistent(self) -> None:
        dev = np.load(self.subvolume_path, allow_pickle=False)
        self.assertEqual(dev["seismic"].shape, dev["positive_mask"].shape)
        self.assertEqual(dev["seismic"].shape, dev["unknown_mask"].shape)
        self.assertEqual(dev["seismic"].shape, dev["verified_background_mask"].shape)
        self.assertEqual(len(dev["time_idx"]), dev["seismic"].shape[0])
        self.assertEqual(len(dev["iline"]), dev["seismic"].shape[1])
        self.assertEqual(len(dev["xline"]), dev["seismic"].shape[2])
        self.assertListEqual(list(dev["iline"][[0, -1]]), list(DEV_BOX["iline"]))
        self.assertListEqual(list(dev["xline"][[0, -1]]), list(DEV_BOX["crossline"]))
        self.assertListEqual(list(dev["time_idx"][[0, -1]]), list(DEV_BOX["time_idx"]))
        positive = dev["positive_mask"]
        unknown = dev["unknown_mask"]
        verified = dev["verified_background_mask"]
        self.assertTrue(positive.any())
        self.assertTrue(unknown.any())
        self.assertTrue(verified.any())
        self.assertFalse(np.any(positive & unknown))
        self.assertFalse(np.any(positive & verified))
        self.assertFalse(np.any(unknown & verified))
        self.assertTrue(np.all(positive | unknown | verified))

    def test_split_manifest_is_development_only_and_group_isolated(self) -> None:
        split_manifest = json.loads(self.split_manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(split_manifest["development_only"])
        self.assertTrue(split_manifest["group_isolated"])
        self.assertEqual(split_manifest["coordinate_order"], ["tline", "iline", "xline"])
        self.assertFalse(split_manifest["frozen_holdout_accessed"])
        blocks = split_manifest["blocks"]
        self.assertEqual([block["name"] for block in blocks], ["fit", "guard", "validation"])
        self.assertEqual([tuple(block["inline"]) for block in blocks], [tuple(v) for v in DEV_SPLIT.values()])
        self.assertGreater(blocks[0]["positive_voxels"], 0)
        self.assertGreater(blocks[2]["positive_voxels"], 0)
        self.assertGreater(blocks[0]["fault_point_count"], 0)
        self.assertGreater(blocks[2]["fault_point_count"], 0)

    def test_manifest_paths_resolve_from_project_root_and_hash_match(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["manifest_path"],
            "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_test/manifest.json",
        )
        for entry in manifest["inputs"] + manifest["outputs"]:
            path = PROJECT_ROOT / entry["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest["status"], "READY")
        self.assertEqual(manifest["reason_code"], "LEGAL_CONTIGUOUS_3D_DEVELOPMENT_VOLUME_READY")


if __name__ == "__main__":
    unittest.main()
