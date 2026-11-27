"""CPU HD1910 P sweep, no RL policy and no connection to physical hardware."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from mjlab_microduck.robot.feetech1910.config import load_config
from mjlab_microduck.robot.feetech1910_test_bench.bench import BenchConfig, build_bench, run_trial


def plot_trials(trials, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    first = next(iter(trials.values()))
    axes[0].plot(first['time_s'], np.degrees(first['target_rad']), 'k--', label='Requested target')
    for p, trace in trials.items():
        t = trace['time_s']
        axes[0].plot(t, np.degrees(trace['position_rad']), label=f'P={p}')
        axes[1].plot(t, np.degrees(trace['velocity_rad_s']), label=f'P={p}')
        axes[2].plot(t, trace['torque_nm'], label=f'P={p}')
    for ax, ylabel in zip(axes, ('Angle (deg)', 'Velocity (deg/s)', 'Motor torque (Nm)')):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend(loc='best', ncols=3)
    axes[2].set_xlabel('Time (s)')
    fig.suptitle('HD1910 step response — shared XL330 fixture CAD, BAM manufacturer prior')
    fig.tight_layout()
    fig.savefig(out / 'response.png', dpi=150)
    plt.close(fig)


def replay(cfg, trace):
    """Replay measured simulation states; viewing never commands a real servo."""
    import mujoco
    import mujoco.viewer

    _, model, data, _, _ = build_bench(cfg)
    print('Replaying saved simulation; close the MuJoCo window to exit.', flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [.1, 0, .2]
        viewer.cam.distance = .65
        while viewer.is_running():
            for i, t in enumerate(trace['time_s']):
                start = time.monotonic()
                if not viewer.is_running():
                    return
                data.time = t
                data.qpos[0] = trace['position_rad'][i]
                data.qvel[0] = trace['velocity_rad_s'][i]
                data.ctrl[0] = trace['torque_nm'][i]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0, cfg.dt - (time.monotonic() - start)))


def main():
    profile = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p', type=int, nargs='+', default=[profile.position_p], help='Firmware P values, e.g. 16 32 64 180')
    parser.add_argument('--voltage', type=float, default=profile.supply_voltage_v)
    parser.add_argument('--initial-deg', type=float, default=0)
    parser.add_argument('--step-deg', type=float, default=5, help='Signed increment from initial angle')
    parser.add_argument('--step-time', type=float, default=.5)
    parser.add_argument('--duration', type=float, default=3)
    parser.add_argument('--dt', type=float, default=.005, help='Physics timestep in seconds')
    parser.add_argument('--delay-steps', type=int, default=0, help='Fixed physics-step delay; 0 isolates the servo, NOT randomized training latency')
    parser.add_argument('--arm-mass', type=float, default=.12, help='Total moving assembly mass in kg; scale CAD inertia at fixed geometry/CoM')
    parser.add_argument('--tolerance-deg', type=float, default=.1, help='Absolute target band for settling time')
    parser.add_argument('--out', type=Path, default=Path('logs/feetech_testbench') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))
    parser.add_argument('--view-p', type=int, help='Replay one tested P in a 3D window (Mac: run with mjpython)')
    args = parser.parse_args()
    gains = list(dict.fromkeys(args.p))
    if args.view_p is not None and args.view_p not in gains:
        parser.error('--view-p must be one of --p')
    try:
        configs = {p: BenchConfig(**{key: getattr(args, key) for key in BenchConfig.__dataclass_fields__ if key != 'p'}, p=p) for p in gains}
    except ValueError as error:
        parser.error(str(error))
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be empty or new to avoid overwriting prior experiments')
    args.out.mkdir(parents=True, exist_ok=True)
    print('Simulation only. Geometry shared with XL330 bench; manufacturer BAM prior, not hardware identification.', flush=True)
    print(f'{args.voltage:g} V, moving mass {args.arm_mass:g} kg, step {args.step_deg:g} deg, dt {args.dt:g} s, delay {args.delay_steps} steps', flush=True)
    trials = {}
    summary = {'fixture': 'xl330_test_bench/scene.xml (shared CAD)', 'model_status': 'manufacturer_prior_not_robot_identified', 'trials': []}
    for p, cfg in configs.items():
        trace = run_trial(cfg)
        trials[p] = trace
        with (args.out / f'p{p}.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            names = ('time_s', 'target_rad', 'applied_target_rad', 'position_rad',
                     'velocity_rad_s', 'torque_nm', 'control_voltage_v')
            writer.writerow(names)
            writer.writerows(zip(*(trace[name] for name in names)))
        summary['trials'].append({'config': asdict(cfg), 'torque_limit_nm': trace['torque_limit_nm'], 'metrics': trace['metrics']})
        m = trace['metrics']
        settling = 'not settled' if m['settling_time_s'] is None else f"{m['settling_time_s']:.3f} s"
        print(f"P={p:3}: final error {m['final_error_deg']:+.3f} deg; overshoot {m['overshoot_deg']:.3f} deg; settling {settling}; voltage saturated {m['voltage_saturation_fraction']:.1%}", flush=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    plot_trials(trials, args.out)
    print(f'Results: {args.out.resolve()}', flush=True)
    if args.view_p is not None:
        replay(configs[args.view_p], trials[args.view_p])


if __name__ == '__main__':
    main()
