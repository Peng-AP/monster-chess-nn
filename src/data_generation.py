import argparse
import json
import os
import time

from tqdm import tqdm

from config import (
    MCTS_SIMULATIONS, NUM_GAMES,
    TEMPERATURE_HIGH, TEMPERATURE_LOW, TEMPERATURE_MOVES,
    RAW_DATA_DIR,
)
from monster_chess import MonsterChessGame
from mcts import MCTS


def play_game(mcts_engine):
    """Play one full game of Monster Chess via MCTS self-play.

    Returns a list of records, one per position:
        {fen, mcts_value, policy, current_player, game_result}
    """
    game = MonsterChessGame()
    records = []
    move_number = 0

    while not game.is_terminal():
        temperature = TEMPERATURE_HIGH if move_number < TEMPERATURE_MOVES else TEMPERATURE_LOW

        action, action_probs, root_value = mcts_engine.get_best_action(
            game, temperature=temperature,
        )

        if action is None:
            break

        is_white = game.is_white_turn
        records.append({
            "fen": game.fen(),
            "mcts_value": round(root_value, 4),
            "policy": action_probs,  # dict: action_str -> visit proportion
            "current_player": "white" if is_white else "black",
        })

        game.apply_action(action)
        move_number += 1

    # Final game result
    result = game.get_result()

    # Stamp every position with the game outcome
    for rec in records:
        rec["game_result"] = result

    return records


def save_game(records, output_dir, game_id):
    """Save a single game's data as a JSON-lines file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"game_{game_id:05d}.jsonl")
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate Monster Chess training data via MCTS self-play")
    parser.add_argument("--num-games", type=int, default=NUM_GAMES)
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS)
    parser.add_argument("--output-dir", type=str, default=RAW_DATA_DIR)
    args = parser.parse_args()

    engine = MCTS(num_simulations=args.simulations)

    total_positions = 0
    results = {1: 0, -1: 0, 0: 0}

    print(f"Generating {args.num_games} games with {args.simulations} MCTS simulations per move...")
    print(f"Output directory: {args.output_dir}\n")

    for i in tqdm(range(args.num_games), desc="Games"):
        t0 = time.time()
        records = play_game(engine)
        elapsed = time.time() - t0

        save_game(records, args.output_dir, game_id=i)

        n_moves = len(records)
        total_positions += n_moves
        game_result = records[-1]["game_result"] if records else 0
        results[game_result] = results.get(game_result, 0) + 1

        winner = {1: "White", -1: "Black", 0: "Draw"}.get(game_result, "?")
        tqdm.write(f"  Game {i}: {n_moves} moves, {winner} wins, {elapsed:.1f}s")

    print(f"\nDone! {total_positions} total positions across {args.num_games} games.")
    print(f"Results — White: {results.get(1,0)}, Black: {results.get(-1,0)}, Draw: {results.get(0,0)}")


if __name__ == "__main__":
    main()
