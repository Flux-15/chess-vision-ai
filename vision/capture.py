"""Screen capture and webcam feed handler for chess vision pipeline."""

from __future__ import annotations
import sys
import ctypes
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

# Enable Windows Per-Monitor DPI Awareness so screen pixel coordinates match accurately
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class FrameCapture:
    """Handles high-performance screen capture or webcam capture."""

    def __init__(
        self,
        source: str = "screen",
        region: Optional[Tuple[int, int, int, int] | list[int]] = None,
        camera_id: int = 0,
    ) -> None:
        """
        Initialize capture source.

        Args:
            source: "screen" for desktop capture or "webcam" for camera feed.
            region: Optional (left, top, width, height) bounding box for screen grab.
            camera_id: Camera index for webcam mode (default 0).
        """
        self.source = source.lower()
        self.region = list(region) if region is not None else None
        self.camera_id = camera_id

        self._sct = None
        self._cam = None

        if self.source == "screen":
            import mss  # Lazy import
            self._sct = mss.mss()
        elif self.source == "webcam":
            self._cam = cv2.VideoCapture(self.camera_id)
            if not self._cam.isOpened():
                raise RuntimeError(f"Could not open webcam with id {self.camera_id}")
        else:
            raise ValueError(f"Unknown capture source: {source}. Choose 'screen' or 'webcam'.")

    def set_region(self, region: Tuple[int, int, int, int] | list[int] | None) -> None:
        """Update region bounding box [left, top, width, height]."""
        self.region = list(region) if region is not None else None

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame as a BGR numpy array.

        Returns:
            np.ndarray: BGR image array (H, W, 3).
        """
        if self.source == "screen":
            if self.region is not None:
                monitor = {
                    "left": int(self.region[0]),
                    "top": int(self.region[1]),
                    "width": int(self.region[2]),
                    "height": int(self.region[3]),
                }
            else:
                # Capture primary monitor
                monitor = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]

            sct_img = self._sct.grab(monitor)
            # mss returns BGRA array, convert to BGR
            frame = np.array(sct_img, dtype=np.uint8)[:, :, :3]
            return frame

        elif self.source == "webcam":
            ret, frame = self._cam.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture frame from webcam.")
            return frame

        raise RuntimeError("Capture device not configured.")

    def release(self) -> None:
        """Release underlying capture resources."""
        if self._cam is not None:
            self._cam.release()
            self._cam = None
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def __enter__(self) -> FrameCapture:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
