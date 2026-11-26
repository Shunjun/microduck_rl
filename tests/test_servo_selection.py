"""The hardware choice and actuator defaults are shared by training and CPU playback."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mjlab_microduck.robot import servo_config


class ServoSelectionTests(unittest.TestCase):
    def test_default_is_feetech(self):
        self.assertEqual(servo_config.SERVO_DRIVER, "feetech1910")
        self.assertEqual(servo_config.actuator_kwargs()["kp_fw"], 180)

    def test_xl330_defaults_are_preserved(self):
        data = servo_config.actuator_kwargs("xl330")
        self.assertEqual(data["motor_name"], "xl330")
        self.assertEqual(data["kp_fw"], 200)
        self.assertEqual(data["vin_range"], (6.5, 8.2))
        self.assertEqual(data["vin_drop_gain_range"], (0.0, 0.2))
        self.assertEqual((data["delay_min_lag"], data["delay_max_lag"]), (3, 6))
        data["kp_fw"] = -1
        self.assertEqual(servo_config.actuator_kwargs("xl330")["kp_fw"], 200)

    def test_feetech_does_not_borrow_xl330_physics(self):
        data = servo_config.actuator_kwargs("feetech1910")
        self.assertTrue(data["json_path"].endswith("feetech1910/reference/m6.json"))
        self.assertEqual(data["kp_fw"], 180)
        self.assertEqual(data["vin"], 7.4)
        self.assertNotIn("vin_drop_gain_range", data)
        self.assertEqual((data["delay_min_lag"], data["delay_max_lag"]), (3, 6))

    def test_invalid_driver_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            servo_config.actuator_kwargs("feetech191")

    def test_cpu_model_uses_the_selected_controller_and_defaults(self):
        ft = servo_config.load_bam_model(driver="feetech1910")
        xl = servo_config.load_bam_model(driver="xl330")
        self.assertEqual(type(ft.actuator).__name__, "HD1910Actuator")
        self.assertEqual(ft.actuator.kp, 180)
        self.assertEqual(ft.actuator.vin, 7.4)
        self.assertEqual(xl.actuator.kp, 200)
        self.assertNotEqual(ft.kt.value, xl.kt.value)


if __name__ == "__main__":
    unittest.main()
