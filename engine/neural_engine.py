"""Custom neural policy/value engine with optional lightweight MCTS.

Architecture:
  - Input: 8x8x(12 piece planes + 5 auxiliary planes) = 17 input channels
  - Residual trunk: configurable depth of residual blocks
  - Policy head: 4096-dimensional output (64*64 from-to move space)
  - Value head: scalar tanh output in [-1, +1] (win/draw/loss)
  - Optional MCTS: PUCT search using policy priors and leaf value evaluation
"""

from __future__ import annotations
import os
import sys
import math
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.base import BaseEngine

logger = logging.getLogger(__name__)

NUM_INPUT_PLANES = 17  # 12 piece planes + side-to-move + 4 castling rights
NUM_MOVES = 4096       # 64 from-squares * 64 to-squares (legal mask applied)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encodes a chess.Board into a (17, 8, 8) float32 tensor.

    Plane layout:
      0-5:   White pieces (P, N, B, R, Q, K)
      6-11:  Black pieces (p, n, b, r, q, k)
      12:    Side to move (1.0 = White to move)
      13:    White can castle kingside
      14:    White can castle queenside
      15:    Black can castle kingside
      16:    Black can castle queenside
    """
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)

    piece_map = board.piece_map()
    piece_to_plane = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square, piece in piece_map.items():
        plane_idx = piece_to_plane.get((piece.piece_type, piece.color))
        if plane_idx is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            planes[plane_idx, 7 - rank, file] = 1.0

    # Auxiliary planes
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    return torch.from_numpy(planes)


def move_to_index(move: chess.Move) -> int:
    """Encodes a chess.Move as a flat index in [0, 4096)."""
    return move.from_square * 64 + move.to_square


def index_to_move_square(idx: int) -> Tuple[int, int]:
    """Decodes a flat index back to (from_square, to_square)."""
    return idx // 64, idx % 64


class ResidualBlock(nn.Module):
    """Single residual block: Conv->BN->ReLU->Conv->BN + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNetwork(nn.Module):
    """
    Dual-headed residual network producing policy logits and value estimate.

    Args:
        num_res_blocks: Number of residual blocks in the trunk (default 6, lightweight).
        channels: Filter count per residual block (default 128).
    """

    def __init__(self, num_res_blocks: int = 6, channels: int = 128) -> None:
        super().__init__()

        # Input projection
        self.input_conv = nn.Conv2d(NUM_INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual trunk
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, NUM_MOVES)

        # Value head
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, 17, 8, 8) board tensor batch.

        Returns:
            policy_logits: (N, 4096) unnormalized move log-probabilities.
            value: (N,) value estimate in [-1, +1].
        """
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


class MCTSNode:
    """Node in the Monte-Carlo Tree Search tree."""

    __slots__ = ["board_fen", "parent", "move", "children", "visit_count",
                 "value_sum", "prior", "is_expanded"]

    def __init__(
        self,
        board_fen: str,
        parent: Optional["MCTSNode"],
        move: Optional[chess.Move],
        prior: float,
    ) -> None:
        self.board_fen = board_fen
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior = prior
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """PUCT upper confidence bound score."""
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


class NeuralEngine(BaseEngine):
    """
    Chess engine backed by the PolicyValueNetwork with optional MCTS search.

    When mcts_simulations > 0, performs PUCT search using the network's policy
    and value outputs. Otherwise uses a single-pass policy argmax (fast but weaker).
    """

    def __init__(
        self,
        weights_path: Optional[str] = "models/policy_value.pt",
        num_res_blocks: int = 6,
        channels: int = 128,
        mcts_simulations: int = 100,
        c_puct: float = 1.5,
        time_budget_ms: int = 2000,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            weights_path: Path to trained model checkpoint.
            num_res_blocks: Residual blocks in the network trunk.
            channels: Filter channels per block.
            mcts_simulations: Number of MCTS simulations per move (0 = greedy policy).
            c_puct: Exploration constant for PUCT formula.
            time_budget_ms: Maximum search time per move in milliseconds.
            device: "cuda", "cpu", or None for auto-detection.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.time_budget_ms = time_budget_ms

        self.network = PolicyValueNetwork(num_res_blocks=num_res_blocks, channels=channels)
        self.network.to(self.device)
        self.network.eval()

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            state = checkpoint.get("model_state", checkpoint)
            self.network.load_state_dict(state)
            logger.info(f"[NeuralEngine] Loaded weights from: {weights_path}")
        else:
            logger.warning(f"[NeuralEngine] No weights found at {weights_path}. Using random network.")

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """
        Runs a forward pass on the board.

        Returns:
            Tuple (policy_probs [4096], value_scalar).
        """
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device.type, enabled=(self.device.type == "cuda")):
            logits, value = self.network(tensor)

        # Mask illegal moves
        legal_indices = {move_to_index(m) for m in board.legal_moves}
        mask = torch.full((NUM_MOVES,), float("-inf"), device=self.device)
        for idx in legal_indices:
            mask[idx] = 0.0

        masked_logits = logits.squeeze(0) + mask
        probs = torch.softmax(masked_logits, dim=0).cpu().numpy()
        return probs, float(value.squeeze().cpu())

    def _mcts_search(self, board: chess.Board, deadline_s: float) -> chess.Move:
        """Runs PUCT MCTS from the current position within the time budget."""
        root = MCTSNode(board.fen(), parent=None, move=None, prior=1.0)
        probs, _ = self._evaluate(board)
        self._expand(root, board, probs)

        for _ in range(self.mcts_simulations):
            if time.time() > deadline_s:
                break
            self._simulate(root)

        # Select move with highest visit count
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        return best_move

    def _expand(self, node: MCTSNode, board: chess.Board, policy_probs: np.ndarray) -> None:
        """Creates child nodes for all legal moves using network policy priors."""
        for move in board.legal_moves:
            idx = move_to_index(move)
            prior = float(policy_probs[idx])
            child_board = board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(
                board_fen=child_board.fen(),
                parent=node,
                move=move,
                prior=prior,
            )
        node.is_expanded = True

    def _simulate(self, root: MCTSNode) -> None:
        """Performs one simulation: selection -> evaluation -> backup."""
        node = root
        path: List[MCTSNode] = [node]

        # Selection: traverse tree following maximum UCB
        while node.is_expanded and node.children:
            parent_visits = node.visit_count
            best_child = max(
                node.children.values(),
                key=lambda c: c.ucb_score(parent_visits, self.c_puct),
            )
            node = best_child
            path.append(node)

        # Evaluation at leaf
        board = chess.Board(node.board_fen)
        if board.is_game_over():
            result = board.result()
            value = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
            if board.turn == chess.BLACK:
                value = -value
        else:
            probs, value = self._evaluate(board)
            if not node.is_expanded:
                self._expand(node, board, probs)

        # Backup: propagate value up the tree
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value  # Flip perspective

    def _greedy_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Fast single-pass policy argmax (no MCTS)."""
        probs, _ = self._evaluate(board)
        legal_map = {move_to_index(m): m for m in board.legal_moves}
        if not legal_map:
            return None
        best_idx = max(legal_map.keys(), key=lambda idx: probs[idx])
        return legal_map[best_idx]

    def best_move(self, board: chess.Board, time_left_ms: int = 2000) -> Optional[chess.Move]:
        """
        Returns best move using MCTS (if configured) or greedy policy argmax.
        """
        if board.is_game_over():
            return None

        if self.mcts_simulations <= 0:
            return self._greedy_move(board)

        think_ms = min(self.time_budget_ms, max(100, time_left_ms // 20))
        deadline = time.time() + (think_ms / 1000.0)
        return self._mcts_search(board, deadline_s=deadline)

    def close(self) -> None:
        """No persistent resources to clean up."""
        pass
