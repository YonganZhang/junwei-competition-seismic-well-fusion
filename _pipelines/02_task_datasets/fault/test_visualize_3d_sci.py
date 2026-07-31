from __future__ import annotations

import unittest

import numpy as np

from visualize_3d_sci import FIELDS, load_context, plane_coordinates


class FaultThreeDSciTests(unittest.TestCase):
    def test_real_spatial_context_contract(self) -> None:
        context = load_context()
        self.assertEqual(FIELDS, ("seismic", "truth", "probability"))
        self.assertEqual(context["patches"].shape, (3, 33, 65))
        self.assertEqual(context["truth"].shape, (3, 33, 65))
        self.assertEqual(context["probability"].shape, (3, 33, 65))
        self.assertEqual(len(set(context["selected_indices"])), 3)
        self.assertTrue(np.isfinite(context["patches"]).all())
        self.assertTrue(np.isfinite(context["probability"]).all())
        self.assertTrue(0.0 < context["time_step_ms"] < 20.0)

        for patch, position in zip(context["patches"], context["positions"]):
            xx, yy, zz = plane_coordinates(position, patch.shape, context["time_step_ms"])
            self.assertEqual(xx.shape, patch.shape)
            self.assertEqual(yy.shape, patch.shape)
            self.assertEqual(zz.shape, patch.shape)
            self.assertTrue(np.all(xx == float(position["inline"])))
            self.assertAlmostEqual(float(np.median(yy)), float(position["crossline"]))
            self.assertAlmostEqual(float(np.median(zz)), float(position["time_ms"]))


if __name__ == "__main__":
    unittest.main()
