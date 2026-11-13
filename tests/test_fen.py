"""Tests for 8x8 grid to FEN conversion and FEN parsing."""

from __future__ import annotations
import pytest
import chess
from vision.state import grid_to_fen_placement, fen_placement_to_grid


def test_starting_position_grid_to_fen() -> None:
    """Tests starting position grid converts accurately to standard starting FEN placement."""
    grid = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]

    fen_placement = grid_to_fen_placement(grid, flipped=False)
    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert fen_placement == expected


def test_flipped_starting_position() -> None:
    """Tests grid captured from Black's perspective converts accurately to standard FEN."""
    # When flipped, row 0 is rank 1 from h to a, row 7 is rank 8 from h to a
    grid_flipped = [
        ["R", "N", "B", "K", "Q", "B", "N", "R"],  # h1, g1, f1, e1, d1, c1, b1, a1
        ["P", "P", "P", "P", "P", "P", "P", "P"],  # h2..a2
        [None] * 8,
        [None] * 8,
        [None] * 8,
        [None] * 8,
        ["p", "p", "p", "p", "p", "p", "p", "p"],  # h7..a7
        ["r", "n", "b", "k", "q", "b", "n", "r"],  # h8..a8
    ]

    fen_placement = grid_to_fen_placement(grid_flipped, flipped=True)
    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert fen_placement == expected


def test_empty_runs_compression() -> None:
    """Tests compression of consecutive empty squares into integers."""
    grid = [[None for _ in range(8)] for _ in range(8)]
    grid[0][0] = "r"
    grid[0][7] = "r"
    grid[7][4] = "K"

    fen_placement = grid_to_fen_placement(grid, flipped=False)
    assert fen_placement == "r6r/8/8/8/8/8/8/4K3"


def test_fen_roundtrip() -> None:
    """Tests that FEN -> grid -> FEN produces identical string."""
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
        "8/8/8/4k3/8/8/4K3/8",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R",
    ]

    for fen in test_positions:
        grid = fen_placement_to_grid(fen)
        reconstructed = grid_to_fen_placement(grid, flipped=False)
        assert reconstructed == fen
