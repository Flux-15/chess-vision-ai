"""Chess board detector and perspective transform (homography) module."""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import cv2


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral coordinates in canonical order:
    [top-left, top-right, bottom-right, bottom-left].

    Args:
        pts: Array of shape (4, 2).

    Returns:
        Ordered array of shape (4, 2) dtype float32.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum (x + y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest diff (y - x) or largest (x - y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


class BoardDetector:
    """Locates the chess board in a frame and warps it to a canonical square image."""

    def __init__(self, target_size: int = 512, min_area_ratio: float = 0.05) -> None:
        """
        Initialize detector.

        Args:
            target_size: Canonical square dimension in pixels (default 512).
            min_area_ratio: Minimum fraction of total frame area to qualify as a board.
        """
        self.target_size = target_size
        self.min_area_ratio = min_area_ratio
        self.last_corners: Optional[np.ndarray] = None

    def detect_grid_board(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Fast and robust detector for axis-aligned chess boards on computer screens.
        Finds bounding boxes with ~1.0 aspect ratio that maximize the 7x7 internal
        checkerboard line gradient energy.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        edges = cv2.Canny(gray, 40, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_score = -1.0
        best_rect: Optional[Tuple[int, int, int, int]] = None

        min_dim = min(h, w)
        min_size = int(min_dim * 0.20)  # Board must be at least 20% of min screen dimension

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < min_size or bh < min_size:
                continue

            aspect = bw / float(bh)
            if not (0.90 <= aspect <= 1.10):
                continue

            sub = gray[by : by + bh, bx : bx + bw]
            if sub.shape[0] < 120 or sub.shape[1] < 120:
                continue

            # Check for the 7 internal horizontal and vertical grid lines
            dy = np.abs(cv2.Sobel(sub, cv2.CV_64F, 0, 1, ksize=3))
            dx = np.abs(cv2.Sobel(sub, cv2.CV_64F, 1, 0, ksize=3))

            step_y = bh / 8.0
            step_x = bw / 8.0
            grid_energy = 0.0

            for k in range(1, 8):
                ry = int(round(k * step_y))
                rx = int(round(k * step_x))
                if 2 <= ry < bh - 2:
                    grid_energy += dy[ry - 2 : ry + 3, :].mean()
                if 2 <= rx < bw - 2:
                    grid_energy += dx[:, rx - 2 : rx + 3].mean()

            score = (bw * bh) * (1.0 + grid_energy)
            if score > best_score:
                best_score = score
                best_rect = (bx, by, bw, bh)

        if best_rect is not None and best_score > 100000.0:
            bx, by, bw, bh = best_rect
            corners = np.array(
                [
                    [bx, by],
                    [bx + bw, by],
                    [bx + bw, by + bh],
                    [bx, by + bh],
                ],
                dtype=np.float32,
            )
            self.last_corners = corners
            return corners

        return None

    def detect_board_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempts to detect board corners automatically via edge and contour analysis.
        First tries axis-aligned grid detection for screens, then contour quad approximation.

        Args:
            frame: Input BGR image.

        Returns:
            Ordered 4 corners np.ndarray of shape (4, 2) or None if not found.
        """
        # Try screen grid detector first (superior precision on desktop monitors)
        grid_corners = self.detect_grid_board(frame)
        if grid_corners is not None:
            return grid_corners

        h, w = frame.shape[:2]
        frame_area = h * w
        min_area = frame_area * self.min_area_ratio

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-scale edge / adaptive threshold detection
        edges = cv2.Canny(blurred, 40, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_corners: Optional[np.ndarray] = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = order_quad_points(approx)

                # Compute width and height of quadrilateral
                w1 = np.linalg.norm(corners[1] - corners[0])
                w2 = np.linalg.norm(corners[2] - corners[3])
                h1 = np.linalg.norm(corners[3] - corners[0])
                h2 = np.linalg.norm(corners[2] - corners[1])

                avg_w = (w1 + w2) / 2.0
                avg_h = (h1 + h2) / 2.0

                if avg_h == 0 or avg_w == 0:
                    continue

                aspect_ratio = avg_w / avg_h
                # Chess board must be roughly square (aspect ratio between 0.85 and 1.15)
                if 0.85 <= aspect_ratio <= 1.15:
                    # Test for checkerboard internal gradient variation (texture score)
                    warped_test = self.warp_from_corners(frame, corners, target_size=128)
                    gray_test = cv2.cvtColor(warped_test, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray_test, cv2.CV_64F).var()

                    score = area * (1.0 + min(laplacian_var / 100.0, 5.0))
                    if score > best_score:
                        best_score = score
                        best_corners = corners

        if best_corners is not None:
            self.last_corners = best_corners
            return best_corners

        return self.last_corners

    def warp_from_corners(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        target_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Applies perspective warp using 4 ordered corner points.

        Args:
            frame: Input image (H, W, 3).
            corners: Ordered 4 corners (top-left, top-right, bottom-right, bottom-left).
            target_size: Target square pixel dimension (defaults to self.target_size).

        Returns:
            np.ndarray: Top-down warped image of shape (target_size, target_size, 3).
        """
        size = target_size or self.target_size
        ordered = order_quad_points(corners)

        dst = np.array(
            [
                [0, 0],
                [size - 1, 0],
                [size - 1, size - 1],
                [0, size - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(frame, matrix, (size, size), flags=cv2.INTER_LINEAR)
        return warped

    def detect_and_warp(
        self,
        frame: np.ndarray,
        fixed_region: Optional[Tuple[int, int, int, int] | list[int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detects board and returns warped top-down image along with corner coordinates.

        Args:
            frame: Input BGR image.
            fixed_region: Optional manual bounding box [x, y, w, h] to use directly.

        Returns:
            Tuple (warped_board, corners) or (None, None) if not found.
        """
        if fixed_region is not None:
            x, y, w, h = [int(v) for v in fixed_region]
            corners = np.array(
                [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h],
                ],
                dtype=np.float32,
            )
            warped = self.warp_from_corners(frame, corners)
            return warped, corners

        corners = self.detect_board_contour(frame)
        if corners is not None:
            warped = self.warp_from_corners(frame, corners)
            return warped, corners

        # If detection fails, use entire frame if roughly square
        h, w = frame.shape[:2]
        if 0.85 <= (w / h) <= 1.15:
            corners = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                dtype=np.float32,
            )
            warped = cv2.resize(frame, (self.target_size, self.target_size))
            return warped, corners

        return None, None
