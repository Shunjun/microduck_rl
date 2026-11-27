"""Sequential P6 training on one GPU; run with --help or --plan first."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("velocity", "sitstand", "standup", "ground_pick", "kick_right", "kick_left", "roulade", "velstand")
TASKS = {
    "velocity": "Mjlab-Velocity-Flat-MicroDuck",
    "sitstand": "Mjlab-SitStand-Flat-MicroDuck",
    "standup": "Mjlab-StandUp-Flat-MicroDuck",
    "ground_pick": "Mjlab-GroundPick-Flat-MicroDuck",
    "roulade": "Mjlab-Roulade-Flat-MicroDuck",
    "kick_left": "Mjlab-BallKick-Left-Flat-MicroDuck",
    "kick_right": "Mjlab-BallKick-Right-Flat-MicroDuck",
}


def remaining_iterations(saved_iteration: int, target: int) -> int:
    """rsl_rl starts at the saved label again; target=4000 ends at model_3999."""
    return 0 if saved_iteration >= target - 1 else target - saved_iteration


def local_teacher_config(base: dict, walk: str, stand: str) -> dict:
    return {
        **base,
        "checkpoint_path": stand,
        "anchor_checkpoint_path": walk,
        "hf_checkpoint": None,
        "anchor_hf_checkpoint": None,
        "wandb_run_path": None,
        "anchor_wandb_run_path": None,
        # The upstream yaw lookup was measured on its XL330 walking policy.
        "anchor_yaw_remap": None,
    }


def checkpoint_iteration(path: Path) -> int:
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=True)["iter"])


def saved_environment(path: Path) -> dict:
    import yaml

    # mjlab writes Python tuple/function tags. BaseLoader reads their data
    # without importing or executing Python objects from the saved YAML.
    env_file = path.parent / "params/env.yaml"
    return yaml.load(env_file.read_text(), Loader=yaml.BaseLoader)


def check_checkpoint_profile(path: Path) -> None:
    config = saved_environment(path)
    actuators = config["scene"]["entities"]["robot"]["articulation"]["actuators"]
    if not actuators or any(float(a.get("kp_fw", -1)) != 6 or "feetech1910" not in a.get("json_path", "") for a in actuators):
        raise ValueError(f"Checkpoint environment is not Feetech P6: {path}")


def worker(args) -> None:
    import torch
    from mjlab.scripts.train import TrainConfig, launch_training
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    stage = args.worker
    task = TASKS.get(stage)
    if stage == "velstand":
        base = "Mjlab-VelStand-Flat-MicroDuck"
        agent = load_rl_cfg(base)
        agent.algorithm.bc_cfg = local_teacher_config(
            agent.algorithm.bc_cfg, str(args.walk_checkpoint), str(args.stand_checkpoint)
        )

        class LocalWalkRunner(MicroduckOnPolicyRunner):
            def __init__(self, *runner_args, **kwargs):
                super().__init__(*runner_args, **kwargs)
                walk = torch.load(args.walk_checkpoint, map_location=self.device, weights_only=True)
                # Transfer the actor/normalizer, keeping a fresh critic, optimizer
                # and curricula for this different task (not a Velocity resume).
                self.alg.actor.load_state_dict(walk["actor_state_dict"], strict=True)
                print("VelStand: actor initialized from local P6 walk; fresh critic and counters", flush=True)

        task = "Mjlab-VelStand-LocalTeachers-Flat-P6-MicroDuck"
        register_mjlab_task(
            task_id=task,
            env_cfg=load_env_cfg(base),
            play_env_cfg=load_env_cfg(base, play=True),
            rl_cfg=agent,
            runner_cls=LocalWalkRunner,
        )

    cfg = TrainConfig.from_task(task)
    cfg.env.scene.num_envs = 64 if args.smoke else args.num_envs
    cfg.agent.max_iterations = 5 if args.smoke else args.iterations
    cfg.agent.logger = "tensorboard"
    cfg.agent.experiment_name = f"batch_p6_{stage}"
    cfg.agent.run_name = f"{args.batch_id}_{'smoke' if args.smoke else 'train'}"
    if stage == "velocity":
        # The P6 run used vx ±0.5; the repository default is now ±0.4.
        # Preserve its saved command ranges when continuing that policy.
        ranges = saved_environment(args.walk_checkpoint)["commands"]["twist"]["ranges"]
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            setattr(cfg.env.commands["twist"].ranges, axis, tuple(map(float, ranges[axis])))
    if args.check_config:
        print(f"Validated {task}: {cfg.agent.experiment_name}, {cfg.env.scene.num_envs} envs, {cfg.agent.max_iterations} iterations")
        return
    if stage == "velocity":
        from mjlab_microduck.checkpoints import INIT_CHECKPOINT_ENV, install_init_checkpoint_patch

        os.environ[INIT_CHECKPOINT_ENV] = str(args.walk_checkpoint)
        install_init_checkpoint_patch()

    log_root = ROOT / "logs/rsl_rl" / cfg.agent.experiment_name
    previous = set(log_root.glob("*"))
    launch_training(task, cfg)
    if not args.smoke:
        checkpoints = [p for d in set(log_root.glob("*")) - previous for p in d.glob("model_*.pt")]
        if not checkpoints:
            raise RuntimeError(f"No checkpoint produced by {stage}")
        latest = max(checkpoints, key=lambda p: int(p.stem.removeprefix("model_")))
        if checkpoint_iteration(latest) < args.target - 1:
            raise RuntimeError(f"{stage} ended before target: {latest}")
        args.result.write_text(json.dumps({"checkpoint": str(latest.resolve())}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk-checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=int, default=4000, help="Total iterations per task, not extra iterations")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--plan", action="store_true", help="Print the queue without training")
    parser.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--stand-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-id", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--check-config", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.target <= 0 or args.num_envs <= 0:
        parser.error("--target and --num-envs must be positive")
    if args.worker:
        worker(args)
        return
    for stage in STAGES:
        print(f"{stage}: {'resume existing PT to' if stage == 'velocity' else 'train'} {args.target} iterations", flush=True)
    if args.plan:
        return

    from mjlab_microduck.robot.servo_config import actuator_kwargs, selected_driver

    if selected_driver() != "feetech1910" or actuator_kwargs()["kp_fw"] != 6:
        raise SystemExit("Expected feetech1910 with P=6")
    walk = args.walk_checkpoint.expanduser().resolve(strict=True)
    # Reject a P8/XL330 checkpoint before an expensive batch starts.
    check_checkpoint_profile(walk)
    batch_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = ROOT / "logs/batches" / batch_id
    batch_dir.mkdir(parents=True)
    stand = None
    for stage in STAGES:
        iterations = remaining_iterations(checkpoint_iteration(walk), args.target) if stage == "velocity" else args.target
        if iterations == 0:
            print("Velocity already reached the target; using existing checkpoint", flush=True)
            continue
        result = batch_dir / f"{stage}.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", stage,
            "--walk-checkpoint", str(walk), "--target", str(args.target),
            "--iterations", str(iterations), "--num-envs", str(args.num_envs),
            "--batch-id", batch_id, "--result", str(result),
        ]
        if stand:
            command += ["--stand-checkpoint", str(stand)]
        env = dict(os.environ, MICRODUCK_WARM_START="0", PYTHONUNBUFFERED="1")
        # Each smoke check runs in a separate process and does not feed the
        # formal run. A failed smoke or formal run stops the whole queue.
        subprocess.run([*command, "--smoke"], cwd=ROOT, env=env, check=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        checkpoint = Path(json.loads(result.read_text())["checkpoint"])
        if stage == "velocity":
            walk = checkpoint
        elif stage == "standup":
            stand = checkpoint
    print(f"Batch finished; checkpoint paths: {batch_dir}", flush=True)


if __name__ == "__main__":
    main()
