"""Templates for visual corruptions applied to raw LIBERO observations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal
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


def make_corruption(
    method: str,
    kernel_size: int = 5,
    camera: CameraName = "image2",
) -> VisualCorruption:
    """Build the corruption selected by runner.py's CLI options."""
    if method == "clean":
        return NoCorruption()
    if method == "blur":
        return GaussianBlur(camera=camera, kernel_size=kernel_size)
    if method == "turnoff":
        return FrameTurnedOff(camera=camera)
    raise ValueError(f"Unknown corruption '{method}'. Choose 'clean', 'blur', or 'turnoff'.")
