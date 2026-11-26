"""Read the original XL330 training settings without importing torch or mjlab."""

import json
from pathlib import Path


def actuator_kwargs() -> dict:
    data = json.loads(Path(__file__).with_name("profile.json").read_text())
    operating = data["operating"]
    return {
        "motor_name": data["bam"]["motor_name"],
        "model": data["bam"]["model"],
        "target_names_expr": (r"^(?!passive_).*",),
        "kp_fw": operating["position_p"],
        "vin_range": tuple(operating["voltage_range_v"]),
        "vin_drop_gain_range": tuple(operating["voltage_drop_gain_range"]),
        "vin_min": operating["voltage_min_v"],
        "delay_min_lag": operating["delay_steps"][0],
        "delay_max_lag": operating["delay_steps"][1],
    }
