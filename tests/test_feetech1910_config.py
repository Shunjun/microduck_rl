"""HD1910 source integrity and deployment-specific configuration gates (CPU only)."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src/mjlab_microduck/robot/feetech1910"
spec = importlib.util.spec_from_file_location("feetech1910_config", PROFILE / "config.py")
config = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = config
spec.loader.exec_module(config)


class Feetech1910ConfigTests(unittest.TestCase):
    def test_operating_point_is_separate_from_manufacturer_reference(self):
        profile = config.load_config()
        self.assertEqual(profile.supply_voltage_v, 7.4)
        self.assertEqual(profile.position_p, 180)
        self.assertEqual(profile.delay_steps, (3, 6))
        self.assertEqual(profile.position_p_address, 21)
        reference = config.reference_parameters()
        self.assertEqual(reference["actuator"], "hd1910")
        self.assertAlmostEqual(reference["kt"], 0.5644993059031307)

    def test_register_21_is_selected_and_missing_delay_is_rejected(self):
        profile = config.load_config()
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError, "delay"):
            replace(profile, delay_steps=None).actuator_kwargs()
        kwargs = replace(profile, delay_steps=(1, 3)).actuator_kwargs()
        self.assertEqual(kwargs["kp_fw"], 180)
        self.assertEqual(kwargs["vin"], 7.4)
        self.assertEqual((kwargs["delay_min_lag"], kwargs["delay_max_lag"]), (1, 3))
        self.assertNotIn("motor_name", kwargs)
        self.assertTrue(Path(kwargs["json_path"]).is_file())

    def test_invalid_operating_points_and_delay_ranges_are_rejected(self):
        from dataclasses import replace
        profile = config.load_config()
        for voltage in (0, 9, float("nan")):
            with self.assertRaises(ValueError):
                replace(profile, supply_voltage_v=voltage)
        for address in (50, 0x32, True):
            with self.assertRaisesRegex(ValueError, "address 21"):
                replace(profile, position_p_address=address)
        for p in (-1, 255, 1.5, True):
            with self.assertRaises(ValueError):
                replace(profile, position_p=p)
        for delay in ((-1, 2), (3, 1), (0.5, 2), (True, 1)):
            with self.assertRaises(ValueError):
                replace(profile, delay_steps=delay)

    def test_catalogue_voltage_and_units_are_preserved(self):
        data = json.loads((PROFILE / "profile.json").read_text())
        point = data["catalogue"]["performance_at_voltage"]
        self.assertEqual(point["voltage_v"], 4.8)
        self.assertEqual(point["stall_torque_kgf_cm"], 9)
        self.assertEqual(point["rated_load_kgf_cm"], 2.2)
        self.assertAlmostEqual(point["stall_torque_nm"], 9 * 0.0980665)
        self.assertNotIn("performance_at_7_4v", data["catalogue"])


if __name__ == "__main__":
    unittest.main()
