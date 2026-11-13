"""PyTorch CNN piece classifier inference module for 13 chess classes."""

from __future__ import annotations
import os
import sys
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision.squares import SquareSplitter, get_square_name
from vision.state import CLASS_TO_PIECE, CLASS_NAMES, grid_to_fen_placement


class PieceClassifier:
    """Classifies 64 chess squares into 13 classes (6 pieces x 2 colors + empty)."""

    def __init__(
        self,
        weights_path: Optional[str] = "models/classifier.pt",
        model_name: str = "mobilenet_v3_small",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize classifier.

        Args:
            weights_path: Path to trained PyTorch weights file.
            model_name: "mobilenet_v3_small" or "resnet18".
            device: "cuda", "cpu", or None for auto-selection.
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name

        # Resolve weights path relative to project root if needed
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resolved_path = weights_path
        if resolved_path and not os.path.isabs(resolved_path):
            candidate = os.path.join(project_root, resolved_path)
            if os.path.exists(candidate):
                resolved_path = candidate
        self.weights_path = resolved_path

        # Normalization constants (ImageNet standard)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.model = self._build_model()
        self.splitter = SquareSplitter(board_size=512, square_size=64, margin_crop=3)

    def _build_model(self) -> nn.Module:
        """Constructs and loads model weights."""
        if self.model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, 13)
        elif self.model_name == "resnet18":
            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 13)
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")

        if self.weights_path and os.path.exists(self.weights_path):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"[*] Loaded trained classifier weights from: {self.weights_path}")
        else:
            print(f"[!] Warning: No weights file found at {self.weights_path}. Model initialized randomly.")

        model.to(self.device)
        model.eval()
        return model

    def preprocess_batch(self, crops: np.ndarray) -> torch.Tensor:
        """
        Converts uint8 BGR crops array (N, H, W, 3) to normalized tensor (N, 3, H, W).
        """
        # Convert BGR to RGB
        crops_rgb = crops[:, :, :, ::-1].copy()
        tensor = torch.from_numpy(crops_rgb).to(self.device).float() / 255.0
        # Permute (N, H, W, C) -> (N, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        # Normalize
        tensor = (tensor - self.mean) / self.std
        return tensor

    @torch.no_grad()
    def predict_crops(self, crops: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs batched inference on an array of square crops (N, H, W, 3).

        Returns:
            Tuple (pred_class_indices [N], probabilities [N, 13])
        """
        tensor = self.preprocess_batch(crops)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_classes = np.argmax(probs, axis=1)
        return pred_classes, probs

    def predict_board(
        self,
        board_img: np.ndarray,
        flipped: bool = False,
    ) -> Dict[str, Any]:
        """
        Splits a 512x512 warped board, classifies all 64 squares in a single batch,
        and returns structured predictions, confidence matrix, and FEN.

        Args:
            board_img: 512x512 BGR warped board image.
            flipped: True if Black is at the bottom.

        Returns:
            Dict containing:
                "fen_placement": string piece placement (e.g. "rnbqkbnr/pppppppp/..."),
                "grid_pieces": 8x8 list of piece chars or None,
                "grid_confidences": 8x8 list of float confidences,
                "square_predictions": dict mapping square names (e.g. "e4") to (piece, conf),
                "flipped": boolean orientation.
        """
        names, crops = self.splitter.get_squares_batch(board_img, flipped=flipped)
        pred_classes, probs = self.predict_crops(crops)

        grid_pieces: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]
        grid_confidences: List[List[float]] = [[0.0 for _ in range(8)] for _ in range(8)]
        square_predictions: Dict[str, Tuple[Optional[str], float]] = {}

        for idx, (name, class_id) in enumerate(zip(names, pred_classes)):
            row = idx // 8
            col = idx % 8
            piece_char = CLASS_TO_PIECE[class_id]
            conf = float(probs[idx, class_id])

            grid_pieces[row][col] = piece_char
            grid_confidences[row][col] = conf
            square_predictions[name] = (piece_char, conf)

        fen_placement = grid_to_fen_placement(grid_pieces, flipped=flipped)

        return {
            "fen_placement": fen_placement,
            "grid_pieces": grid_pieces,
            "grid_confidences": grid_confidences,
            "square_predictions": square_predictions,
            "flipped": flipped,
        }
