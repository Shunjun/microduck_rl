"""Batch ordering and zero-based checkpoint iteration budgets."""

import importlib.util
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_remaining.py"
spec = importlib.util.spec_from_file_location("train_remaining", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class TrainingBatchTests(unittest.TestCase):
    def test_tagged_mjlab_yaml_is_read_without_loading_python_objects(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "params").mkdir()
            config = root / "params/env.yaml"
            config.write_text('scene:\n  entities:\n    robot:\n      articulation:\n        actuators: !!python/tuple\n        - kp_fw: 6.0\n          json_path: robot/feetech1910/reference/m6.json\n')
            batch.check_checkpoint_profile(root / "model_1999.pt")
            config.write_text(config.read_text().replace('kp_fw: 6.0', 'kp_fw: 8.0'))
            with self.assertRaisesRegex(ValueError, 'not Feetech P6'):
                batch.check_checkpoint_profile(root / "model_1999.pt")

    def test_resume_reaches_target_checkpoint_without_extra_4000_iterations(self):
        # rsl_rl restarts its loop at the saved iteration, repeating that label.
        self.assertEqual(batch.remaining_iterations(1999, 4000), 2001)
        self.assertEqual(batch.remaining_iterations(3999, 4000), 0)
        self.assertEqual(batch.remaining_iterations(4999, 4000), 0)

    def test_velstand_has_local_teachers_and_no_upstream_yaw_calibration(self):
        cfg = batch.local_teacher_config({"anchor_yaw_remap": {"old": True}}, "walk.pt", "stand.pt")
        self.assertEqual(cfg["checkpoint_path"], "stand.pt")
        self.assertEqual(cfg["anchor_checkpoint_path"], "walk.pt")
        self.assertIsNone(cfg["anchor_yaw_remap"])
        self.assertIsNone(cfg["hf_checkpoint"])
        self.assertIsNone(cfg["anchor_hf_checkpoint"])

    def test_teachers_precede_velstand_and_both_kicking_feet_are_included(self):
        self.assertLess(batch.STAGES.index("velocity"), batch.STAGES.index("velstand"))
        self.assertLess(batch.STAGES.index("standup"), batch.STAGES.index("velstand"))
        self.assertIn("kick_left", batch.STAGES)
        self.assertIn("kick_right", batch.STAGES)
        self.assertFalse(any("roller" in stage for stage in batch.STAGES))
