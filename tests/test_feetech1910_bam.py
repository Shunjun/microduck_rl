"""Manufacturer HD1910 controller against the BAM revision pinned by this repo."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("bam"), "requires the locked BAM dependency")
class Feetech1910BamTests(unittest.TestCase):
    def test_reference_loads_with_its_own_controller_and_preserves_xl330(self):
        from bam.model import load_model
        from mjlab_microduck.robot.feetech1910.config import load_reference_model

        xl330_before = load_model(motor_name="xl330", model="m6")
        model = load_reference_model()
        self.assertEqual(model.actuator_name, "hd1910")
        self.assertEqual(type(model.actuator).__name__, "HD1910Actuator")
        self.assertEqual(model.actuator.kp, 32)
        self.assertEqual(model.actuator.vin, 7.4)
        self.assertAlmostEqual(model.kt.value, 0.5644993059031307)
        self.assertAlmostEqual(model.armature.value, 0.0018273861091867637)
        self.assertTrue(math.isfinite(model.actuator.compute_torque(1.0, True, 0.0, 0.0)))
        xl330_after = load_model(motor_name="xl330", model="m6")
        self.assertEqual(type(xl330_before.actuator), type(xl330_after.actuator))
        self.assertEqual(xl330_before.kt.value, xl330_after.kt.value)

    def test_reference_control_law_and_voltage_saturation(self):
        from mjlab_microduck.robot.feetech1910.config import load_reference_model

        model = load_reference_model()
        actuator = model.actuator
        self.assertAlmostEqual(actuator.compute_control(0.01, 0.0, 0.0, 0.004), 7.4 * 32 * 0.163 * 0.01)
        self.assertAlmostEqual(actuator.compute_control(2.0, 0.0, 0.0, 0.004), 7.4 * 0.97)
        self.assertAlmostEqual(actuator.compute_control(-2.0, 0.0, 0.0, 0.004), -7.4 * 0.97)


if __name__ == "__main__":
    unittest.main()
