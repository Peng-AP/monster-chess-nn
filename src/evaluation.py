import chess
import numpy as np


def evaluate(game_state):
    """Evaluate a Monster Chess position heuristically.

    Monster Chess specifics:
      - White king gets 2 moves/turn, so a lone king can still hunt
        and capture Black's king. White without pawns is NOT lost.
      - Black wins by promoting pawns into heavy pieces (Q/R) and
        building a mating barrier to push White's king to the edge.
      - White wins by advancing pawns to promote, or capturing Black's king
        directly with the double-move advantage.

    Returns a float in [-1, 1] from White's perspective.
    Designed to be swappable with a neural-network evaluation later.
    """
    board = game_state.board

    # Terminal
    if board.king(chess.WHITE) is None:
        return -1.0
    if board.king(chess.BLACK) is None:
        return 1.0

    score = 0.0

    # ---- White's chances ----

    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    score += white_pawns * 0.10

    # White pawn advancement
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        score += (rank - 1) * 0.05

    # White queens (promoted pawns)
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    score += white_queens * 0.30

    # ---- Black's chances ----

    # Black heavy pieces (queens + rooks) — needed for mating net
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
    black_heavy = black_queens + black_rooks
    score -= black_heavy * 0.08

    # Black pawn advancement (pawns promote on rank 0 for Black)
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        rank = chess.square_rank(sq)
        score -= (6 - rank) * 0.03

    # Black promoted queens beyond starting one — strong signal
    if black_queens > 1:
        score -= (black_queens - 1) * 0.35

    # ---- King confinement (helps Black's mating pattern) ----
    wk = board.king(chess.WHITE)
    if wk is not None and black_heavy >= 2:
        wk_rank = chess.square_rank(wk)
        wk_file = chess.square_file(wk)

        # Edge proximity bonus — king on edge is easier to mate
        rank_edge = min(wk_rank, 7 - wk_rank)
        file_edge = min(wk_file, 7 - wk_file)
        edge_dist = min(rank_edge, file_edge)
        score -= (3 - edge_dist) * 0.10

        # Adjacent squares attacked by Black — measures king confinement
        adjacent_attacked = 0
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                r, f = wk_rank + dr, wk_file + df
                if 0 <= r <= 7 and 0 <= f <= 7:
                    sq = chess.square(f, r)
                    if board.is_attacked_by(chess.BLACK, sq):
                        adjacent_attacked += 1
        # Strong signal: more attacked adjacent squares = more confined
        score -= adjacent_attacked * 0.06

        # Bonus for heavy pieces controlling same rank/file as king
        for sq in board.pieces(chess.QUEEN, chess.BLACK) | board.pieces(chess.ROOK, chess.BLACK):
            sq_rank = chess.square_rank(sq)
            sq_file = chess.square_file(sq)
            if sq_rank == wk_rank or sq_file == wk_file:
                score -= 0.08  # piece directly aiming at king's rank/file

    # ---- Material balance (general) ----
    black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    score -= (black_knights + black_bishops) * 0.03

    return max(-0.95, min(0.95, score))


class NNEvaluator:
    """Neural network evaluator that wraps a trained Keras model.

    Converts game_state -> tensor -> model.predict -> scalar value.
    Batches predictions for efficiency.
    """

    def __init__(self, model_path):
        from tensorflow import keras
        from data_processor import fen_to_tensor
        self.model = keras.models.load_model(model_path)
        self.fen_to_tensor = fen_to_tensor

    def __call__(self, game_state):
        board = game_state.board

        # Terminal states — don't need the model
        if board.king(chess.WHITE) is None:
            return -1.0
        if board.king(chess.BLACK) is None:
            return 1.0

        tensor = self.fen_to_tensor(
            game_state.fen(),
            is_white_turn=game_state.is_white_turn,
        )
        # model expects batch dimension
        pred = self.model(np.expand_dims(tensor, axis=0), training=False)
        return float(pred[0, 0])

    def batch_evaluate(self, game_states):
        """Evaluate multiple game states in a single model call."""
        results = []
        indices_to_predict = []
        tensors = []

        for i, gs in enumerate(game_states):
            board = gs.board
            if board.king(chess.WHITE) is None:
                results.append(-1.0)
            elif board.king(chess.BLACK) is None:
                results.append(1.0)
            else:
                results.append(None)  # placeholder
                indices_to_predict.append(i)
                tensors.append(self.fen_to_tensor(
                    gs.fen(), is_white_turn=gs.is_white_turn,
                ))

        if tensors:
            batch = np.stack(tensors, axis=0)
            preds = self.model(batch, training=False).numpy().flatten()
            for j, idx in enumerate(indices_to_predict):
                results[idx] = float(preds[j])

        return results
