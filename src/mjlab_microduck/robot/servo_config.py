"""One hardware selection for training and CPU playback; imports no torch/mjlab."""

# Change only this constant to switch the robot's servo model.
SERVO_DRIVER = "feetech1910"  # "xl330" | "feetech1910"


def selected_driver(driver: str | None = None) -> str:
    name = SERVO_DRIVER if driver is None else driver
    if name not in ("xl330", "feetech1910"):
        raise ValueError(f"Unknown SERVO_DRIVER: {name!r}; use xl330 or feetech1910")
    return name


def actuator_kwargs(driver: str | None = None) -> dict:
    if selected_driver(driver) == "xl330":
        from .xl330.config import actuator_kwargs as xl330_kwargs
        return xl330_kwargs()
    from .feetech1910.config import load_config
    return load_config().actuator_kwargs()


def prepare_bam_registry(driver: str | None = None) -> None:
    if selected_driver(driver) == "feetech1910":
        from .feetech1910.config import load_reference_model
        load_reference_model()


def load_bam_model(kp_fw=None, vin=None, max_current=None, *, driver: str | None = None):
    """CPU model using the same profile/controller as the mjlab training adapter."""
    name = selected_driver(driver)
    kwargs = actuator_kwargs(name)
    if name == "feetech1910":
        from .feetech1910.config import load_reference_model
        model = load_reference_model()
    else:
        from bam.model import load_model
        model = load_model(motor_name=kwargs["motor_name"], model=kwargs["model"])
    model.actuator.kp = kwargs["kp_fw"] if kp_fw is None else kp_fw
    model.actuator.vin = kwargs.get("vin", 7.4) if vin is None else vin
    model.actuator.max_current = max_current if max_current is not None and max_current > 0 else None
    return model
