"""Supervised training of the PolicyValueNetwork on the Lichess open-game database.

Usage:
    python training/train_policy.py \\
        --pgn-dir data/pgn \\
        --epochs 10 \\
        --batch-size 256 \\
        --min-elo 1800

This script:
  1. Streams PGN files from --pgn-dir (supports .pgn and .pgn.bz2).
  2. Filters to rated standard games where both players are >= --min-elo.
  3. Builds board tensors and (move_index, game_outcome) labels.
  4. Trains PolicyValueNetwork with:
       - Policy loss: cross-entropy over legal-move-masked 4096-dim output
       - Value loss: MSE between predicted value and actual game outcome (-1/0/+1)
  5. Evaluates on a held-out validation split and saves best checkpoint.

Data download:
    Download from https://database.lichess.org/ and place .pgn.bz2 files in data/pgn/.
"""

from __future__ import annotations
import os
import sys
import bz2
import time
import json
import logging
import argparse
from typing import Iterator, List, Tuple, Optional, Dict, Any

import numpy as np
import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.neural_engine import PolicyValueNetwork, board_to_tensor, move_to_index, NUM_MOVES

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _open_pgn(path: str):
    """Opens a .pgn or .pgn.bz2 file for reading."""
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_elo(game: chess.pgn.Game) -> Tuple[int, int]:
    """Returns (white_elo, black_elo) from PGN headers, or (0, 0) if missing."""
    try:
        white_elo = int(game.headers.get("WhiteElo", "0") or "0")
        black_elo = int(game.headers.get("BlackElo", "0") or "0")
        return white_elo, black_elo
    except ValueError:
        return 0, 0


def _game_result_to_value(result: str) -> float:
    """
    Converts PGN result string to a scalar value from White's perspective.

    Returns:
        1.0 for White win, -1.0 for Black win, 0.0 for draw or unknown.
    """
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    return 0.0


def stream_positions(
    pgn_dir: str,
    min_elo: int = 1800,
    max_positions_per_file: int = 50000,
) -> Iterator[Tuple[chess.Board, int, float]]:
    """
    Streams (board, move_index, value) triples from PGN files.

    Args:
        pgn_dir: Directory containing .pgn or .pgn.bz2 files.
        min_elo: Minimum Elo for both players (filter low-quality games).
        max_positions_per_file: Cap on positions extracted from each file.

    Yields:
        Tuple of (chess.Board at a position, move_index played, game outcome value
                 from the perspective of the side to move).
    """
    if not os.path.isdir(pgn_dir):
        logger.error(f"PGN directory not found: {pgn_dir}")
        return

    pgn_files = [
        os.path.join(pgn_dir, f)
        for f in sorted(os.listdir(pgn_dir))
        if f.endswith(".pgn") or f.endswith(".pgn.bz2")
    ]

    if not pgn_files:
        logger.warning(f"No PGN files found in {pgn_dir}.")
        return

    logger.info(f"Found {len(pgn_files)} PGN file(s) in {pgn_dir}.")

    for pgn_path in pgn_files:
        count = 0
        logger.info(f"  Processing: {os.path.basename(pgn_path)}")
        try:
            with _open_pgn(pgn_path) as f:
                while count < max_positions_per_file:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    white_elo, black_elo = _parse_elo(game)
                    if min(white_elo, black_elo) < min_elo:
                        continue

                    variant = game.headers.get("Variant", "Standard")
                    if variant.lower() not in ("standard", ""):
                        continue

                    result_str = game.headers.get("Result", "*")
                    game_value = _game_result_to_value(result_str)
                    if game_value == 0.0 and result_str not in ("1/2-1/2", "1-0", "0-1"):
                        continue  # Skip incomplete games

                    board = game.board()
                    for move in game.mainline_moves():
                        if move not in board.legal_moves:
                            break

                        move_idx = move_to_index(move)
                        # Value from the perspective of side to move
                        value = game_value if board.turn == chess.WHITE else -game_value

                        yield board.copy(), move_idx, value
                        board.push(move)
                        count += 1

                        if count >= max_positions_per_file:
                            break

        except Exception as e:
            logger.warning(f"  Error processing {pgn_path}: {e}")

        logger.info(f"  Extracted {count} positions from {os.path.basename(pgn_path)}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChessPositionDataset(Dataset):
    """In-memory dataset of board tensors, policy labels, and value labels."""

    def __init__(
        self,
        boards: np.ndarray,
        move_indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """
        Args:
            boards: Float32 array of shape (N, 17, 8, 8).
            move_indices: Int64 array of shape (N,) with flat move indices.
            values: Float32 array of shape (N,) with game outcomes in [-1, +1].
        """
        self.boards = boards
        self.move_indices = move_indices
        self.values = values

    def __len__(self) -> int:
        return len(self.move_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        return (
            torch.from_numpy(self.boards[idx]),
            int(self.move_indices[idx]),
            float(self.values[idx]),
        )


def build_dataset(
    pgn_dir: str,
    min_elo: int = 1800,
    max_total_positions: int = 500000,
) -> ChessPositionDataset:
    """
    Builds an in-memory ChessPositionDataset from PGN files.

    Args:
        pgn_dir: Directory of PGN files.
        min_elo: Minimum Elo filter.
        max_total_positions: Cap on total positions loaded.

    Returns:
        ChessPositionDataset instance.
    """
    boards_list: List[np.ndarray] = []
    move_indices_list: List[int] = []
    values_list: List[float] = []

    for board, move_idx, value in stream_positions(pgn_dir, min_elo=min_elo):
        boards_list.append(board_to_tensor(board).numpy())
        move_indices_list.append(move_idx)
        values_list.append(value)

        if len(move_indices_list) >= max_total_positions:
            break

    if not boards_list:
        raise RuntimeError(
            f"No positions loaded from {pgn_dir}. "
            "Download Lichess PGN data from https://database.lichess.org/ "
            "and place .pgn or .pgn.bz2 files in that directory."
        )

    boards = np.stack(boards_list, axis=0)
    move_indices = np.array(move_indices_list, dtype=np.int64)
    values = np.array(values_list, dtype=np.float32)

    logger.info(f"Dataset built: {len(move_indices):,} positions from {pgn_dir}")
    return ChessPositionDataset(boards, move_indices, values)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_policy(
    pgn_dir: str = "data/pgn",
    min_elo: int = 1800,
    max_positions: int = 500000,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 2e-4,
    num_res_blocks: int = 6,
    channels: int = 128,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    output_path: str = "models/policy_value.pt",
    report_path: str = "models/policy_training_report.json",
    val_split: float = 0.1,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main training loop for the PolicyValueNetwork.

    Returns:
        Dict with training metrics summary.
    """
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training device: {device}")

    # Build dataset
    dataset = build_dataset(pgn_dir, min_elo=min_elo, max_total_positions=max_positions)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    logger.info(f"Train: {n_train:,} positions | Val: {n_val:,} positions")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0)

    model = PolicyValueNetwork(num_res_blocks=num_res_blocks, channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_top1 = 0
        total = 0
        t0 = time.time()

        for boards, move_idxs, values in train_loader:
            boards = boards.to(device, non_blocking=True)
            move_idxs = move_idxs.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                policy_logits, value_preds = model(boards)
                policy_loss = policy_criterion(policy_logits, move_idxs)
                value_loss = value_criterion(value_preds, values)
                loss = policy_weight * policy_loss + value_weight * value_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_policy_loss += policy_loss.item() * boards.size(0)
            total_value_loss += value_loss.item() * boards.size(0)
            preds = policy_logits.argmax(dim=1)
            correct_top1 += (preds == move_idxs).sum().item()
            total += boards.size(0)

        scheduler.step()
        train_policy_loss = total_policy_loss / total
        train_value_loss = total_value_loss / total
        train_top1 = correct_top1 / total

        # Validation
        model.eval()
        val_policy_loss_total = 0.0
        val_value_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for boards, move_idxs, values in val_loader:
                boards = boards.to(device, non_blocking=True)
                move_idxs = move_idxs.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True).float()

                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    policy_logits, value_preds = model(boards)
                    p_loss = policy_criterion(policy_logits, move_idxs)
                    v_loss = value_criterion(value_preds, values)

                val_policy_loss_total += p_loss.item() * boards.size(0)
                val_value_loss_total += v_loss.item() * boards.size(0)
                val_correct += (policy_logits.argmax(dim=1) == move_idxs).sum().item()
                val_total += boards.size(0)

        val_policy_loss = val_policy_loss_total / val_total
        val_value_loss = val_value_loss_total / val_total
        val_top1 = val_correct / val_total
        val_combined = policy_weight * val_policy_loss + value_weight * val_value_loss

        elapsed = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_policy_loss": round(train_policy_loss, 4),
            "train_value_loss": round(train_value_loss, 4),
            "train_top1_acc": round(train_top1 * 100, 2),
            "val_policy_loss": round(val_policy_loss, 4),
            "val_value_loss": round(val_value_loss, 4),
            "val_top1_acc": round(val_top1 * 100, 2),
            "time_s": round(elapsed, 1),
        }
        history.append(entry)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"TrainPoL={train_policy_loss:.4f} TrainValL={train_value_loss:.4f} TrainTop1={train_top1*100:.1f}% | "
            f"ValPoL={val_policy_loss:.4f} ValValL={val_value_loss:.4f} ValTop1={val_top1*100:.1f}% | "
            f"Time={elapsed:.1f}s"
        )

        if val_combined < best_val_loss:
            best_val_loss = val_combined
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "num_res_blocks": num_res_blocks,
                "channels": channels,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, output_path)
            logger.info(f"  -> New best checkpoint saved to {output_path}")

    report = {"history": history, "best_val_loss": best_val_loss, "best_epoch": min(range(len(history)), key=lambda i: history[i]["val_policy_loss"]) + 1}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training complete. Best checkpoint: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Policy/Value Network on Lichess PGN data")
    parser.add_argument("--pgn-dir", type=str, default="data/pgn", help="Directory containing .pgn or .pgn.bz2 files")
    parser.add_argument("--min-elo", type=int, default=1800, help="Minimum player Elo filter")
    parser.add_argument("--max-positions", type=int, default=500000, help="Maximum positions to load")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--res-blocks", type=int, default=6, help="Residual blocks in network")
    parser.add_argument("--channels", type=int, default=128, help="Filter channels per block")
    parser.add_argument("--output", type=str, default="models/policy_value.pt", help="Checkpoint output path")
    args = parser.parse_args()

    train_policy(
        pgn_dir=args.pgn_dir,
        min_elo=args.min_elo,
        max_positions=args.max_positions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_res_blocks=args.res_blocks,
        channels=args.channels,
        output_path=args.output,
    )
