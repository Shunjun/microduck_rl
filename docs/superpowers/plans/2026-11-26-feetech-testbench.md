# Feetech testbench implementation plan

**Goal:** Provide a CPU-only, policy-free HD1910 step-response bench to compare P gains.

**Architecture:** Reuse the original XL330 fixture CAD and meshes (user confirmed the servo geometry is the same), existing HD1910 BAM controller and parameters, a Python rollout module, and a CLI producing CSV, PNG and JSON. Keep existing XL330 tests and runtime configuration unchanged.

**Scope:** Single joint, initial hold then one angular step; independently simulate each requested P at the same voltage, load and initial conditions. Defaults: selected HD1910 profile P/voltage, 5 degree step, original CAD with 120 g moving arm assembly (same mass scaling as XL330). Report signed final error, overshoot, settling time, peak torque and voltage saturation fraction. No automatic choice of a best gain, no hardware writes, no RL task registration. Command delay is an explicit fixed physics-step count, default 0 for isolated servo response; it is not the training's randomized end-to-end latency.

- [x] Add failing tests for step timing, signed responses/settling metrics, independent gain trials, torque limits and validation.
- [x] Reuse the fixture and add CPU BAM rollout, using identical actuator limits/armature/friction stiffness to training.
- [x] Add CLI for P sweep, load, delay, CSV/JSON and PNG outputs, plus optional native 3D replay.
- [x] Run tests and sample P=16/32/64/180 sweeps; verify numerical output and inspect the rendered plot.
- [x] Document commands and limits; do not commit or launch services.
