"""Templates for visual corruptions applied to raw LIBERO observations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence
import cv2
import numpy as np


CameraName = Literal["image", "image2"]
RawObservation = dict[str, Any]


class VisualCorruption(ABC):
    """Common interface that lets runner.py execute any camera corruption."""

    name: str

    def reset(self) -> None:
        """Clear episode-specific state before every rollout."""

    @abstractmethod
    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        """Return an observation with the same keys, shapes, and dtypes."""


@dataclass
class NoCorruption(VisualCorruption):
    """Clean baseline: use the real frames unchanged."""

    name: str = "clean"

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        return raw_observation


@dataclass
class FrameFreeze(VisualCorruption):
    """Template: freeze one camera between start_step and end_step."""

    camera: CameraName
    start_step: int
    duration: int
    name: str = "freeze"
    frozen_frame: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        self.frozen_frame = None

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        # TODO:
        # 1. Read raw_observation["pixels"][self.camera].
        # 2. At start_step, store a .copy() in self.frozen_frame.
        # 3. During the next `duration` steps, replace only this camera frame.
        # 4. Return the observation. Preserve batch/H/W/C shape and uint8 dtype.
        raise NotImplementedError


@dataclass
class FrameRateFreeze(VisualCorruption):
    """Refresh one camera every ``period`` control steps, holding old frames between."""

    camera: CameraName = "image2"
    period: int = 5
    frozen_frame: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError("period must be at least 1")

    @property
    def name(self) -> str:
        camera_index = {"image": 0, "image2": 1}[self.camera]
        return f"freeze_{self.period}_{camera_index}"

    def reset(self) -> None:
        self.frozen_frame = None

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        frames = raw_observation["pixels"][self.camera]
        if self.frozen_frame is None or step % self.period == 0:
            self.frozen_frame = frames.copy()
        else:
            raw_observation["pixels"][self.camera] = self.frozen_frame.copy()
        return raw_observation


@dataclass
class GaussianBlur(VisualCorruption):
    """Apply Gaussian blur to one camera at every control step."""

    camera: CameraName = "image2"
    kernel_size: int = 5

    def __post_init__(self) -> None:
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number, for example 5")

    @property
    def name(self) -> str:
        camera_index = {"image": 0, "image2": 1}[self.camera]
        return f"blur_{self.kernel_size}_{camera_index}"

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        frames = raw_observation["pixels"][self.camera]
        raw_observation["pixels"][self.camera] = np.stack(
            [cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), sigmaX=0) for frame in frames]
        )

        return raw_observation


@dataclass
class FrameTurnedOff(VisualCorruption):
    """Replace one camera with black frames while retaining the expected input shape."""

    camera: CameraName = "image2"

    @property
    def name(self) -> str:
        camera_index = {"image": 0, "image2": 1}[self.camera]
        return f"turnoff_{camera_index}"

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        frames = raw_observation["pixels"][self.camera]
        raw_observation["pixels"][self.camera] = np.zeros_like(frames)
        return raw_observation


@dataclass
class TemporaryFrameTurnedOff(VisualCorruption):
    """Replace one camera with black frames during one bounded interval."""

    camera: CameraName
    start_step: int
    duration: int

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        if self.duration < 1:
            raise ValueError("duration must be positive")

    @property
    def name(self) -> str:
        camera_index = {"image": 0, "image2": 1}[self.camera]
        return f"temporary_off_{self.start_step}_{self.duration}_{camera_index}"

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        if self.start_step <= step < self.start_step + self.duration:
            frames = raw_observation["pixels"][self.camera]
            raw_observation["pixels"][self.camera] = np.zeros_like(frames)
        return raw_observation


@dataclass
class ScheduledFrameTurnedOff(VisualCorruption):
    """Apply multiple camera outages at configured control-step intervals."""

    schedule: Sequence[dict[str, Any]]
    name: str = "scheduled_turnoff"

    def __post_init__(self) -> None:
        if not self.schedule:
            raise ValueError("schedule must contain at least one outage")
        for outage in self.schedule:
            if outage.get("camera") not in {"image", "image2"}:
                raise ValueError(f"Unknown camera in schedule: {outage.get('camera')!r}")
            if not isinstance(outage.get("start_step"), int) or outage["start_step"] < 0:
                raise ValueError("Each start_step must be a non-negative integer")
            if not isinstance(outage.get("duration"), int) or outage["duration"] < 1:
                raise ValueError("Each duration must be a positive integer")

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        for outage in self.schedule:
            start_step = outage["start_step"]
            if start_step <= step < start_step + outage["duration"]:
                camera = outage["camera"]
                frames = raw_observation["pixels"][camera]
                raw_observation["pixels"][camera] = np.zeros_like(frames)
        return raw_observation


@dataclass
class RightEdgeOcclusion(VisualCorruption):
    """Hide a fixed fraction of one camera with a black mask from the right edge."""

    camera: CameraName
    fraction: float

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")

    @property
    def name(self) -> str:
        camera_index = {"image": 0, "image2": 1}[self.camera]
        return f"right_occlusion_{round(self.fraction * 100)}_{camera_index}"

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        frames = raw_observation["pixels"][self.camera].copy()
        mask_width = max(1, round(frames.shape[2] * self.fraction))
        # LIBERO's processor flips both image axes before SmolVLA sees the frame.
        # Mask raw left so the model-oriented image is occluded from the right.
        frames[:, :, :mask_width, :] = 0
        raw_observation["pixels"][self.camera] = frames
        return raw_observation


@dataclass
class CompositeCorruption(VisualCorruption):
    """Apply several camera corruptions to the same raw observation."""

    corruptions: list[VisualCorruption]
    name: str

    def reset(self) -> None:
        for corruption in self.corruptions:
            corruption.reset()

    def apply(self, raw_observation: RawObservation, step: int) -> RawObservation:
        for corruption in self.corruptions:
            raw_observation = corruption.apply(raw_observation, step)
        return raw_observation


def make_corruption(
    method: str,
    kernel_size: int = 5,
    camera: CameraName | Sequence[CameraName] = "image2",
    period: int = 1,
    start_step: int = 0,
    duration: int = 1,
    schedule: Sequence[dict[str, Any]] | None = None,
    periods: dict[CameraName, int] | None = None,
    fraction: float = 0.25,
    name: str | None = None,
) -> VisualCorruption:
    """Build the corruption selected by runner.py's CLI options."""
    if method == "clean":
        return NoCorruption()
    if method == "scheduled_turnoff":
        return ScheduledFrameTurnedOff(schedule=schedule or [], name=name or method)
    if method == "multi_freeze":
        if not periods:
            raise ValueError("multi_freeze requires a non-empty periods mapping")
        unknown_cameras = set(periods) - {"image", "image2"}
        if unknown_cameras:
            raise ValueError(f"Unknown cameras in periods: {sorted(unknown_cameras)}")
        corruptions = [FrameRateFreeze(camera=item, period=value) for item, value in periods.items()]
        return CompositeCorruption(corruptions=corruptions, name=name or "multi_freeze")
    if not isinstance(camera, str):
        cameras = list(camera)
        if not cameras:
            raise ValueError("At least one camera must be selected")
        if len(set(cameras)) != len(cameras):
            raise ValueError("Each camera may appear only once")
        camera_indices = "_".join(str({"image": 0, "image2": 1}[item]) for item in cameras)
        individual = [make_corruption(method, kernel_size, item, period) for item in cameras]
        if method == "blur":
            name = f"blur_{kernel_size}_{camera_indices}"
        elif method == "freeze":
            name = f"freeze_{period}_{camera_indices}"
        else:
            name = f"{method}_{camera_indices}"
        return CompositeCorruption(corruptions=individual, name=name)
    if method == "blur":
        return GaussianBlur(camera=camera, kernel_size=kernel_size)
    if method == "turnoff":
        return FrameTurnedOff(camera=camera)
    if method == "temporary_off":
        return TemporaryFrameTurnedOff(camera=camera, start_step=start_step, duration=duration)
    if method == "freeze":
        return FrameRateFreeze(camera=camera, period=period)
    if method == "right_occlusion":
        return RightEdgeOcclusion(camera=camera, fraction=fraction)
    raise ValueError(
        f"Unknown corruption '{method}'. Choose 'clean', 'blur', 'freeze', 'turnoff', "
        "'temporary_off', 'scheduled_turnoff', 'multi_freeze', or 'right_occlusion'."
    )
