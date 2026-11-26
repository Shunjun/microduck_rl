"""HD1910 configuration, with manufacturer priors separate from measured robot values.

Importing this module does not import torch/mjlab or change the XL330 task defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


PROFILE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = PROFILE_DIR / "reference"


def reference_parameters() -> dict:
    """Read the pinned, unmodified manufacturer fit after checking its provenance."""
    provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
    for name, entry in provenance["files"].items():
        if hashlib.sha256((REFERENCE_DIR / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Manufacturer reference checksum mismatch: {name}")
    parameters = json.loads((REFERENCE_DIR / "m6.json").read_text())
    if parameters.get("actuator") != "hd1910" or parameters.get("model") != "m6":
        raise ValueError("Expected the manufacturer HD1910 M6 reference")
    return parameters


@dataclass(frozen=True, kw_only=True)
class Feetech1910Config:
    supply_voltage_v: float = 7.4
    position_p: int = 180
    position_p_address: int = 21
    # User-selected initial assumption: reuse XL330 3..6 PHYSICS steps.
    # Not measured HD1910/Orange Pi latency; do not treat this as manufacturer data.
    delay_steps: tuple[int, int] | None = (3, 6)

    def __post_init__(self) -> None:
        if not math.isfinite(self.supply_voltage_v) or not 4.0 <= self.supply_voltage_v <= 8.4:
            raise ValueError("HD1910 supply voltage must be within 4.0..8.4 V")
        if type(self.position_p) is not int or not 0 <= self.position_p <= 254:
            raise ValueError("HD1910 position P must be an integer within 0..254")
        if type(self.position_p_address) is not int or self.position_p_address != 21:
            raise ValueError("This HD1910 profile uses position P address 21 (0x15)")
        if self.delay_steps is not None:
            if (
                len(self.delay_steps) != 2
                or any(type(step) is not int for step in self.delay_steps)
                or not 0 <= self.delay_steps[0] <= self.delay_steps[1]
            ):
                raise ValueError("delay_steps must be an ordered pair of nonnegative integers")

    def actuator_kwargs(self) -> dict:
        """Settings for the existing BAM adapter once the unknowns are confirmed.

        P=180 is the user's runtime setting, not a conversion of XL330 kp=200.
        The reference controller was supplied with kp=32 and error_gain=0.163;
        P address 21 (0x15) is selected by the user for the real driver.
        The manufacturer model remains a prior, not a measured P=180 response.
        """
        if self.delay_steps is None:
            raise ValueError("Set measured total delay_steps for the simulation timestep first")
        reference_parameters()
        return {
            "json_path": str(REFERENCE_DIR / "m6.json"),
            "target_names_expr": (r"^(?!passive_).*",),
            "vin": self.supply_voltage_v,
            "kp_fw": float(self.position_p),
            "delay_min_lag": self.delay_steps[0],
            "delay_max_lag": self.delay_steps[1],
        }


def load_config(path: Path | str = PROFILE_DIR / "profile.json") -> Feetech1910Config:
    data = json.loads(Path(path).read_text())
    if data["model"] != "HD-1910-C001":
        raise ValueError("Expected an HD-1910-C001 profile")
    operating = dict(data["operating"])
    if operating.get("delay_steps") is not None:
        operating["delay_steps"] = tuple(operating["delay_steps"])
    return Feetech1910Config(**operating)


def load_reference_model():
    """Load the manufacturer's CPU BAM reference at its own kp=32, vin=7.4 V."""
    reference_parameters()
    from bam import model as bam_model
    from bam.testbench import Pendulum
    from .reference.actuator import HD1910Actuator

    # //Compatible: the locked Rhoban BAM revision has no HD1910 registry entry.
    # Register only that model; preserve XL330 and refuse a different controller.
    if "hd1910" not in bam_model.actuators:
        bam_model.actuators["hd1910"] = lambda: HD1910Actuator(Pendulum)
    reference = bam_model.load_model(str(REFERENCE_DIR / "m6.json"))
    actuator = reference.actuator
    if not (
        type(actuator).__name__ == "HD1910Actuator"
        and math.isclose(actuator.error_gain, 0.163)
        and math.isclose(actuator.max_pwm, 0.97)
    ):
        raise ValueError("The BAM registry contains a different HD1910 controller")
    return reference


def make_actuator_cfg(profile: Feetech1910Config | None = None, *, backlash: bool = False):
    """Build an opt-in actuator cfg; does not replace any registered task's robot.

    Includes the actual HD1910 controller, not XL330's controller with new numbers.
    """
    kwargs = (profile or load_config()).actuator_kwargs()
    load_reference_model()
    from mjlab_microduck.actuator import (
        BacklashEncoderBamActuatorCfg,
        FrictionDRBamActuatorCfg,
    )

    cls = BacklashEncoderBamActuatorCfg if backlash else FrictionDRBamActuatorCfg
    return cls(**kwargs)
