"""
Hybrid chess piece classifier:
  Primary:  CNN (MobileNetV3-Small) trained on ocean theme + glass pieces
  Pre-filter: Color-based occupancy detection (fast, pre-checks empty squares)
  Fallback: Template NCC if CNN confidence below threshold

This module is a drop-in replacement for PieceClassifier that adds the
color-based empty-square optimization and adaptive background calibration.
"""

from __future__ import annotations
import os, sys
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision.state import CLASS_TO_PIECE, CLASS_NAMES, grid_to_fen_placement
from vision.squares import SquareSplitter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OCEAN_LIGHT_BGR = np.array([126, 169, 187], dtype=np.float32)
OCEAN_DARK_BGR  = np.array([ 63, 113, 149], dtype=np.float32)

PIECE_FILES = {
    "wP":"wp","wN":"wn","wB":"wb","wR":"wr","wQ":"wq","wK":"wk",
    "bP":"bp","bN":"bn","bB":"bb","bR":"br","bQ":"bq","bK":"bk",
}
WHITE_PIECES = ["wP","wN","wB","wR","wQ","wK"]
BLACK_PIECES = ["bP","bN","bB","bR","bQ","bK"]


class TemplateClassifier:
    """
    Chess piece classifier with CNN + color-feature pre-filter.

    Uses MobileNetV3-Small CNN (trained on ocean/glass data) as primary recognizer.
    Adds fast occupancy + white/black color pre-filter to skip CNN on empty squares
    and narrow the prediction space.
    """

    def __init__(
        self,
        square_size: int = 64,
        assets_dir: Optional[str] = None,
        light_bgr: Optional[np.ndarray] = None,
        dark_bgr:  Optional[np.ndarray] = None,
        ncc_threshold: float = 0.60,
        cnn_fallback: bool = True,       # kept for API compat — CNN is primary here
        cnn_weights:  Optional[str] = None,
    ) -> None:
        self.sq     = square_size
        self.thresh = ncc_threshold

        if assets_dir is None:
            assets_dir = os.path.join(PROJECT_ROOT, "assets", "pieces")
        self.assets_dir = assets_dir

        self.light_bgr = (light_bgr if light_bgr is not None
                          else OCEAN_LIGHT_BGR.copy()).astype(np.float32)
        self.dark_bgr  = (dark_bgr  if dark_bgr  is not None
                          else OCEAN_DARK_BGR.copy()).astype(np.float32)

        # Build template references for occupancy + NCC fallback
        self.templates: Dict[str, Dict[str, np.ndarray]] = {"light": {}, "dark": {}}
        self._build_templates()

        # Primary CNN classifier
        self.cnn: Optional[Any] = None
        try:
            from vision.classifier import PieceClassifier
            self.cnn = PieceClassifier(weights_path=cnn_weights)
            print("[TemplateClassifier] CNN (primary) loaded.")
        except Exception as exc:
            print(f"[TemplateClassifier] CNN unavailable — will use template-only: {exc}")

        self.splitter = SquareSplitter(board_size=512, square_size=64, margin_crop=0)
        self._calibrated = False

    # ── Template construction ─────────────────────────────────────────────────

    def _composite(self, sprite: Image.Image, bg: np.ndarray) -> np.ndarray:
        r, g, b = int(bg[2]), int(bg[1]), int(bg[0])
        canvas = Image.new("RGB", sprite.size, (r, g, b))
        canvas.paste(sprite, (0, 0), sprite)
        return cv2.cvtColor(np.array(canvas, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    def _build_templates(self) -> None:
        glass_dir = os.path.join(self.assets_dir, "glass")
        for code, fname in PIECE_FILES.items():
            path = os.path.join(glass_dir, f"{fname}.png")
            if not os.path.exists(path):
                path = os.path.join(self.assets_dir, "neo", f"{fname}.png")
            if not os.path.exists(path):
                continue
            with Image.open(path) as img:
                spr = img.convert("RGBA").resize((self.sq, self.sq), Image.LANCZOS)
            self.templates["light"][code] = self._composite(spr, self.light_bgr)
            self.templates["dark"][code]  = self._composite(spr, self.dark_bgr)
        self._empty_light = np.full((self.sq, self.sq, 3),
                                    self.light_bgr.astype(np.uint8), dtype=np.uint8)
        self._empty_dark  = np.full((self.sq, self.sq, 3),
                                    self.dark_bgr.astype(np.uint8),  dtype=np.uint8)
        print(f"[TemplateClassifier] {len(self.templates['light'])} piece templates built.")

    def _rebuild(self) -> None:
        self._build_templates()

    # ── Occupancy + color helpers ─────────────────────────────────────────────

    def _is_occupied(self, crop: np.ndarray, is_light: bool) -> Tuple[bool, float]:
        ref  = self._empty_light if is_light else self._empty_dark
        diff = cv2.absdiff(crop.astype(np.float32), ref.astype(np.float32))
        score = float(diff.mean()) / 255.0
        return score > 0.025, score

    def _piece_color(self, crop: np.ndarray, is_light: bool) -> str:
        """
        White glass pieces are brighter than the board; black glass pieces darker.
        """
        ref = self.light_bgr if is_light else self.dark_bgr
        ref_gray = float(0.114 * ref[0] + 0.587 * ref[1] + 0.299 * ref[2])
        margin = self.sq // 6
        c_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[margin:-margin, margin:-margin]
        return "white" if float(c_gray.mean()) > ref_gray else "black"

    @staticmethod
    def _ncc_color(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for ch in range(3):
            ac = a[:, :, ch].astype(np.float32); bc = b[:, :, ch].astype(np.float32)
            am = ac - ac.mean(); bm = bc - bc.mean()
            total += float(np.sum(am * bm) / (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-9))
        return total / 3.0

    def _template_classify(self, crop: np.ndarray, is_light: bool) -> Tuple[str, float]:
        """Full template-only classification pipeline (no CNN)."""
        occupied, diff = self._is_occupied(crop, is_light)
        if not occupied:
            return "empty", max(0.72, 1.0 - diff * 10.0)
        sq_key = "light" if is_light else "dark"
        color  = self._piece_color(crop, is_light)
        candidates = WHITE_PIECES if color == "white" else BLACK_PIECES
        best_score = -2.0; best_code = candidates[0]
        for code in candidates:
            tmpl = self.templates[sq_key].get(code)
            if tmpl is None:
                continue
            s = self._ncc_color(crop, tmpl)
            if s > best_score:
                best_score = s; best_code = code
        return best_code, float((best_score + 1.0) / 2.0)

    # ── Adaptive calibration ──────────────────────────────────────────────────

    def calibrate_from_board(self, board_img: np.ndarray) -> None:
        """Estimates light/dark board colors from the most uniform squares."""
        light_s: List[np.ndarray] = []
        dark_s:  List[np.ndarray] = []
        candidates: List[Tuple[float, int, int, bool]] = []
        sq = 64
        for r in range(8):
            for c in range(8):
                crop = board_img[r*sq:(r+1)*sq, c*sq:(c+1)*sq]
                if crop.size == 0:
                    continue
                std = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std())
                candidates.append((std, r, c, (r + c) % 2 == 0))
        candidates.sort()
        lf = df = 0
        for std, r, c, is_light in candidates:
            crop = board_img[r*sq:(r+1)*sq, c*sq:(c+1)*sq]
            mean = crop.mean(axis=(0, 1))
            if is_light and lf < 6:
                light_s.append(mean); lf += 1
            elif not is_light and df < 6:
                dark_s.append(mean);  df += 1
            if lf >= 6 and df >= 6:
                break
        changed = False
        if len(light_s) >= 2:
            nl = np.mean(light_s, axis=0).astype(np.float32)
            if np.max(np.abs(nl - self.light_bgr)) > 15:
                self.light_bgr = nl; changed = True
        if len(dark_s) >= 2:
            nd = np.mean(dark_s, axis=0).astype(np.float32)
            if np.max(np.abs(nd - self.dark_bgr)) > 15:
                self.dark_bgr = nd;  changed = True
        if changed:
            print(f"[TemplateClassifier] Calibrated: "
                  f"light={self.light_bgr.astype(int)} dark={self.dark_bgr.astype(int)}")
            self._rebuild()
        self._calibrated = True

    # ── Board classification ──────────────────────────────────────────────────

    def predict_board(
        self,
        board_img: np.ndarray,
        flipped: bool = False,
    ) -> Dict[str, Any]:
        """
        Classifies all 64 squares.

        Flow:
         1. Calibrate board colors (first call only).
         2. Quick occupancy filter — mark empty squares immediately.
         3. Run CNN on all occupied squares in one batched forward pass.
         4. For CNN results below threshold, fall back to template NCC.
        """
        if not self._calibrated:
            self.calibrate_from_board(board_img)

        names, crops = self.splitter.get_squares_batch(board_img, flipped=flipped)

        grid_pieces:       List[List[Optional[str]]] = [[None]*8 for _ in range(8)]
        grid_confidences:  List[List[float]]          = [[0.0]*8  for _ in range(8)]
        square_preds:      Dict[str, Tuple[Optional[str], float]] = {}

        # Stage 1: quick empty-square detection (saves CNN time)
        occupied_indices: List[int] = []
        for idx, (name, crop) in enumerate(zip(names, crops)):
            row = idx // 8; col = idx % 8
            is_light = (row + col) % 2 == 0
            occupied, diff = self._is_occupied(crop, is_light)
            if not occupied:
                conf = max(0.72, 1.0 - diff * 10.0)
                grid_pieces[row][col]     = None
                grid_confidences[row][col] = conf
                square_preds[name]         = (None, conf)
            else:
                occupied_indices.append(idx)

        # Stage 2: CNN on occupied squares
        if occupied_indices and self.cnn is not None:
            occ_crops = crops[occupied_indices]
            cnn_cls, cnn_probs = self.cnn.predict_crops(occ_crops)

            still_uncertain: List[int] = []
            for i, orig in enumerate(occupied_indices):
                row = orig // 8; col = orig % 8
                name = names[orig]
                is_light = (row + col) % 2 == 0
                cid  = int(cnn_cls[i])
                conf = float(cnn_probs[i, cid])
                char = CLASS_TO_PIECE[cid]

                grid_pieces[row][col]     = char
                grid_confidences[row][col] = conf
                square_preds[name]         = (char, conf)

                if conf < self.thresh:
                    still_uncertain.append(orig)

            # Stage 3: template NCC for uncertain occupied squares
            for orig in still_uncertain:
                row = orig // 8; col = orig % 8
                name = names[orig]
                crop = crops[orig]
                is_light = (row + col) % 2 == 0
                code, conf = self._template_classify(crop, is_light)
                # Keep whichever gives higher confidence
                if conf > grid_confidences[row][col]:
                    char = self._code_to_char(code)
                    grid_pieces[row][col]     = char
                    grid_confidences[row][col] = conf
                    square_preds[name]         = (char, conf)

        elif occupied_indices:
            # No CNN — template only
            for orig in occupied_indices:
                row = orig // 8; col = orig % 8
                name = names[orig]
                crop = crops[orig]
                is_light = (row + col) % 2 == 0
                code, conf = self._template_classify(crop, is_light)
                char = self._code_to_char(code)
                grid_pieces[row][col]     = char
                grid_confidences[row][col] = conf
                square_preds[name]         = (char, conf)

        fen = grid_to_fen_placement(grid_pieces, flipped=flipped)
        return {
            "fen_placement":      fen,
            "grid_pieces":        grid_pieces,
            "grid_confidences":   grid_confidences,
            "square_predictions": square_preds,
            "flipped":            flipped,
        }

    @staticmethod
    def _code_to_char(code: str) -> Optional[str]:
        if code == "empty":
            return None
        return code[1].upper() if code[0] == "w" else code[1].lower()
