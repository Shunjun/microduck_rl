"""Physical and measurement checks for the policy-free HD1910 bench."""

import unittest
from dataclasses import replace

import numpy as np
import mujoco
from pathlib import Path

from mjlab_microduck.robot.feetech1910_test_bench.bench import (
    BenchConfig,
    build_bench,
    response_metrics,
    run_trial,
)


class BenchTests(unittest.TestCase):
    def test_shared_cad_and_explicit_hd1910_controller(self):
        cfg = BenchConfig(p=64, arm_mass=.24)
        bam, model, _, _, _ = build_bench(cfg)
        original = mujoco.MjModel.from_xml_path(str(
            Path(__file__).resolve().parents[1] / 'src/mjlab_microduck/robot/xl330_test_bench/scene.xml'
        ))
        self.assertEqual(bam.actuator_name, 'hd1910')
        self.assertEqual(bam.actuator.kp, 64)
        self.assertEqual(model.nmesh, original.nmesh)
        np.testing.assert_allclose(model.mesh_vert, original.mesh_vert)
        np.testing.assert_allclose(model.body('arm').ipos, original.body('arm').ipos)
        self.assertAlmostEqual(model.body('arm').mass.item(), .24)
        np.testing.assert_allclose(model.body('arm').inertia, original.body('arm').inertia * 2.4)
        self.assertAlmostEqual(model.dof_armature[0], bam.armature.value)

    def test_metrics_require_staying_inside_band_and_keep_error_sign(self):
        t = np.arange(6, dtype=float)
        # Two isolated visits to target must not count as settling.
        for sign in (1, -1):
            q = sign * np.array([0, 1, 1.3, 1, 0.7, 0.9])
            metrics = response_metrics(t, q, step_time=1, initial=0, target=sign, tolerance=0.05)
            self.assertIsNone(metrics['settling_time_s'])
            self.assertAlmostEqual(metrics['overshoot_deg'], np.degrees(0.3))
            self.assertAlmostEqual(metrics['final_error_deg'], np.degrees(sign * 0.1))
        settled = response_metrics(t, np.array([0, .5, .9, 1, 1.01, 1]),
                                   step_time=1, initial=0, target=1, tolerance=.05)
        self.assertEqual(settled['settling_time_s'], 2)

    def test_invalid_inputs_fail_before_simulation(self):
        for kwargs in ({'p': 255}, {'voltage': float('nan')}, {'dt': 0},
                       {'duration': .2, 'step_time': .5}, {'delay_steps': -1},
                       {'delay_steps': 1.5}, {'arm_mass': -1},
                       {'step_deg': 0}, {'initial_deg': 79, 'step_deg': 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BenchConfig(**kwargs)

    def test_real_bam_step_timing_limits_and_gain_effect(self):
        cfg = BenchConfig(p=16, dt=.005, duration=.5, step_time=.1, delay_steps=3)
        low = run_trial(cfg)
        high = run_trial(replace(cfg, p=180))
        self.assertTrue(np.isfinite(high['position_rad']).all())
        at = np.flatnonzero(high['target_rad'] != 0)[0]
        applied = np.flatnonzero(high['applied_target_rad'] != 0)[0]
        self.assertAlmostEqual(high['time_s'][at], cfg.step_time)
        self.assertEqual(applied - at, 3)
        self.assertTrue(np.all(np.abs(high['torque_nm']) <= high['torque_limit_nm'] + 1e-9))
        self.assertGreater(np.max(np.abs(high['torque_nm'])), np.max(np.abs(low['torque_nm'])))
        # Fresh trials must not share controller state.
        again = run_trial(cfg)
        np.testing.assert_allclose(low['position_rad'], again['position_rad'])


if __name__ == '__main__':
    unittest.main()
