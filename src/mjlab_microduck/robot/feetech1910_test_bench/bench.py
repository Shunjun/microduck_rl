"""Fixed-base, single-axis BAM/MuJoCo experiment, independent of RL policy.

The fixture meshes/geometry are shared with the original XL330 bench. Controller and motor
parameters come from the same HD1910 loader as training. This isolates the servo;
it does not reproduce the whole robot, UART or training domain randomization.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.robot.servo_config import load_bam_model


@dataclass(frozen=True, kw_only=True)
class BenchConfig:
    p: int = 180
    voltage: float = 7.4
    initial_deg: float = 0.0
    step_deg: float = 5.0
    step_time: float = 0.5
    duration: float = 3.0
    dt: float = 0.005
    delay_steps: int = 0
    arm_mass: float = 0.12
    tolerance_deg: float = 0.1

    def __post_init__(self):
        if type(self.p) is not int or not 0 <= self.p <= 254:
            raise ValueError('P must be an integer in 0..254')
        if type(self.delay_steps) is not int or self.delay_steps < 0:
            raise ValueError('delay_steps must be a nonnegative integer')
        for name in ('voltage', 'initial_deg', 'step_deg', 'step_time', 'duration',
                     'dt', 'arm_mass', 'tolerance_deg'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if not 4 <= self.voltage <= 8.4:
            raise ValueError('Voltage must be in 4..8.4 V')
        if not 0 <= self.step_time < self.duration or self.dt <= 0:
            raise ValueError('Require 0 <= step_time < duration and dt > 0')
        if self.step_time + (self.delay_steps + 2) * self.dt > self.duration:
            raise ValueError('Duration must allow samples after the delayed step')
        if self.arm_mass <= 0:
            raise ValueError('Moving arm assembly mass must be positive')
        if self.step_deg == 0 or self.tolerance_deg <= 0:
            raise ValueError('Step must be nonzero and tolerance must be positive')
        if max(abs(self.initial_deg), abs(self.initial_deg + self.step_deg)) >= 80:
            raise ValueError('Initial and target angles must be strictly inside +/-80 deg')


def build_bench(cfg: BenchConfig):
    from bam.mujoco import MujocoController

    bam = load_bam_model(driver='feetech1910', kp_fw=cfg.p, vin=cfg.voltage)
    # Shared CAD as requested: do not copy meshes or silently replace the fixture.
    source = Path(__file__).resolve().parent.parent / 'xl330_test_bench' / 'scene.xml'
    spec = mujoco.MjSpec.from_file(str(source))
    arm = spec.body('arm')
    scale = cfg.arm_mass / arm.mass
    arm.mass = cfg.arm_mass
    arm.fullinertia = [value * scale for value in arm.fullinertia]
    spec.option.timestep = cfg.dt
    joint = spec.joint('1')
    joint.armature = bam.actuator.get_extra_inertia()
    joint.damping = np.zeros((3, 1))
    joint.frictionloss = 0
    # Same native-friction constraint stiffness and motor ceiling as BamActuator.edit_spec.
    joint.solref_friction = [-5e4, -2e2]
    joint.solimp_friction = [.99, .9999, .001, .5, 2]
    limit = cfg.voltage * bam.kt.value / bam.R.value
    motor = spec.actuator('1')
    motor.set_to_motor()
    motor.ctrllimited = False
    motor.forcelimited = True
    motor.forcerange = [-limit, limit]
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[0] = math.radians(cfg.initial_deg)
    mujoco.mj_forward(model, data)
    controller = MujocoController(bam, ['1'], model, data)
    controller.reset(data.qpos.copy())
    return bam, model, data, controller, limit


def response_metrics(time, position, *, step_time, initial, target, tolerance):
    """Metrics start at the requested step (therefore include configured latency).

    Settling means remaining inside an absolute angular band through the END
    of this recorded window, not a promise of stability beyond that window.
    """
    mask = np.asarray(time) >= step_time - 1e-10
    t = np.asarray(time)[mask] - step_time
    q = np.asarray(position)[mask]
    direction = np.sign(target - initial)
    tail = q[-max(1, len(q) // 5):]
    outside = np.flatnonzero(np.abs(q - target) > tolerance)
    settled = int(outside[-1] + 1) if len(outside) else 0
    progress = direction * (q - initial) / abs(target - initial)
    low, high = np.flatnonzero(progress >= .1), np.flatnonzero(progress >= .9)
    return {
        'final_error_deg': float(np.degrees(target - q[-1])),
        'tail_mean_error_deg': float(np.degrees(target - np.mean(tail))),
        'tail_peak_to_peak_deg': float(np.degrees(np.ptp(tail))),
        'overshoot_deg': float(np.degrees(max(0, np.max(direction * (q - target))))),
        'rise_time_10_90_s': float(t[high[0]] - t[low[0]]) if len(low) and len(high) else None,
        'settling_time_s': float(t[settled]) if settled < len(t) else None,
    }


def run_trial(cfg: BenchConfig) -> dict:
    bam, model, data, controller, limit = build_bench(cfg)
    count = math.ceil(cfg.duration / cfg.dt) + 1
    times = np.arange(count) * cfg.dt
    initial = math.radians(cfg.initial_deg)
    target = math.radians(cfg.initial_deg + cfg.step_deg)
    requested = np.where(times >= cfg.step_time - 1e-10, target, initial)
    delayed = requested[np.maximum(0, np.arange(count) - cfg.delay_steps)]
    position, velocity, torque, voltage = (np.empty(count) for _ in range(4))
    for i in range(count):
        position[i], velocity[i] = data.qpos[0], data.qvel[0]
        controller.q_target[:] = delayed[i]
        controller.update()
        voltage[i] = float(np.asarray(bam.actuator.compute_control(
            controller.q_target, data.qpos[:1], data.qvel[:1], cfg.dt
        )).item())
        torque[i] = np.clip(data.ctrl[0], -limit, limit)
        if i < count - 1:
            mujoco.mj_step(model, data)
    if not all(np.isfinite(x).all() for x in (position, velocity, torque, voltage)):
        raise RuntimeError('Non-finite bench response; inspect timestep/model parameters')
    metrics = response_metrics(times, position, step_time=cfg.step_time,
                               initial=initial, target=target,
                               tolerance=math.radians(cfg.tolerance_deg))
    post = times >= cfg.step_time - 1e-10
    metrics['peak_abs_torque_nm'] = float(np.max(np.abs(torque[post])))
    metrics['voltage_saturation_fraction'] = float(np.mean(
        np.abs(voltage[post]) >= cfg.voltage * bam.actuator.max_pwm - 1e-9))
    return {'time_s': times, 'target_rad': requested, 'applied_target_rad': delayed,
            'position_rad': position, 'velocity_rad_s': velocity, 'torque_nm': torque,
            'control_voltage_v': voltage, 'torque_limit_nm': limit, 'metrics': metrics}
