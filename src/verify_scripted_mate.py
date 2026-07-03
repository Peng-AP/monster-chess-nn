"""Verify ScriptedMate: Black (scripted) vs a real White-playing MCTS AI.

Black must win every game — the algorithm claims K+Q+R+R vs bare double-move
king is a forced win, so a single loss/draw is a defect.

    py -3 src/verify_scripted_mate.py --games 12 --white-sims 400
"""
import argparse
import random

import chess

from monster_chess import MonsterChessGame
from mcts import MCTS
from evaluation import evaluate
from scripted_mate import ScriptedMate


def random_start_fen(rng):
    """K+Q+R+R (Black) vs K (White): White low, Black pieces high and safe."""
    wk = chess.square(rng.randrange(8), rng.randrange(0, 3))
    used = {wk}

    def place(rank_lo, rank_hi):
        while True:
            sq = chess.square(rng.randrange(8), rng.randrange(rank_lo, rank_hi + 1))
            if sq not in used and _cheb(sq, wk) >= 3:
                used.add(sq)
                return sq

    board = chess.Board(None)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(place(6, 7), chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(place(5, 7), chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(place(5, 7), chess.Piece(chess.ROOK, chess.BLACK))
    board.set_piece_at(place(5, 7), chess.Piece(chess.ROOK, chess.BLACK))
    board.turn = chess.WHITE
    return board.fen()


def _cheb(a, b):
    return max(abs(chess.square_rank(a) - chess.square_rank(b)),
               abs(chess.square_file(a) - chess.square_file(b)))


def play_one(fen, white_sims, max_turns=120, verbose=False):
    game = MonsterChessGame(fen=fen)
    white = MCTS(num_simulations=white_sims, eval_fn=evaluate,
                 root_noise=False, allow_early_stop=True)
    black = ScriptedMate()
    plies = 0
    while not game.is_terminal() and game.turn_count < max_turns:
        if game.is_white_turn:
            action, _p, _v = white.get_best_action(game, temperature=0.0)
            if action is None:
                break
            if verbose:
                half = "2" if game.white_half_pending else "1"
                print(f"  t{game.turn_count:3d} W{half} {action.uci()}")
            game.apply_search_action(action)
        else:
            move = black.select_move(game)
            if move is None:
                break
            legal = game.get_search_actions()
            if move not in legal:
                if verbose:
                    print(f"    scripted move {move.uci()} not legal at "
                          f"{game.fen()}; falling back")
                move = legal[0]
            if verbose:
                print(f"  t{game.turn_count:3d} B  {move.uci()}   {game.fen()}")
            game.apply_search_action(move)
        plies += 1
    result = game.get_result() if game.is_terminal() else None
    return result, game.turn_count, game.fen()


def main():
    ap = argparse.ArgumentParser(description="Verify scripted K+Q+R+R vs K mate")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--white-sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fen", type=str, default=None,
                    help="Replay a single start FEN (with --verbose for a trace)")
    args = ap.parse_args()

    if args.fen:
        result, turns, final_fen = play_one(args.fen, args.white_sims,
                                            max_turns=args.max_turns,
                                            verbose=args.verbose)
        print(f"result={result} turns={turns} final={final_fen}")
        return

    rng = random.Random(args.seed)
    wins = 0
    failures = []
    for i in range(args.games):
        fen = random_start_fen(rng)
        random.seed(args.seed * 1000 + i)  # MCTS uses global random: make
        result, turns, final_fen = play_one(fen, args.white_sims,  # runs reproducible
                                            max_turns=args.max_turns,
                                            verbose=args.verbose)
        ok = result is not None and result < 0
        wins += ok
        status = "BLACK WIN" if ok else f"FAIL (result={result})"
        print(f"Game {i:2d}: {status:20s} turns={turns:3d}  start={fen}")
        if not ok:
            failures.append((fen, final_fen, result, turns))

    print(f"\n{wins}/{args.games} Black wins")
    for fen, final_fen, result, turns in failures:
        print(f"  FAILED start: {fen}\n         final: {final_fen} "
              f"(result={result}, turns={turns})")


if __name__ == "__main__":
    main()
