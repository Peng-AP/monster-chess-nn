import chess
import numpy as np


def _white_can_capture_king(board):
    """Check if White can capture Black's king in one double-move.

    Returns True if any (m1, m2) sequence reaches the Black king square.
    Short-circuits on first hit — fast O(m) best case.
    """
    bk = board.king(chess.BLACK)
    if bk is None:
        return False
    wk = board.king(chess.WHITE)
    if wk is None:
        return False

    saved_turn = board.turn
    board.turn = chess.WHITE
    for m1 in board.pseudo_legal_moves:
        if m1.to_square == bk:
            board.turn = saved_turn
            return True
        board.push(m1)
        board.turn = chess.WHITE
        for m2 in board.pseudo_legal_moves:
            if m2.to_square == bk:
                board.pop()
                board.turn = saved_turn
                return True
        board.turn = chess.BLACK
        board.pop()
    board.turn = saved_turn
    return False


def _is_passed_pawn(board, sq, color):
    """Check if a pawn has no opposing pawns blocking or flanking its path."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    opp = not color
    if color == chess.WHITE:
        check_ranks = range(rank + 1, 8)
    else:
        check_ranks = range(0, rank)
    for f in (file - 1, file, file + 1):
        if 0 <= f <= 7:
            for r in check_ranks:
                p = board.piece_at(chess.square(f, r))
                if p and p.piece_type == chess.PAWN and p.color == opp:
                    return False
    return True


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

    # ---- Capture threat scan (dominates all other terms) ----
    if _white_can_capture_king(board):
        return 0.95  # White captures Black's king next turn

    score = 0.0
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    wk_rank = chess.square_rank(wk)
    wk_file = chess.square_file(wk)
    bk_rank = chess.square_rank(bk)
    bk_file = chess.square_file(bk)

    # ---- White's chances ----

    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    score += white_pawns * 0.10

    # White pawn advancement with passed pawn bonus
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        score += (rank - 1) * 0.05
        if _is_passed_pawn(board, sq, chess.WHITE):
            # Exponential bonus for passed pawns (Stockfish pattern)
            passed_bonus = 0.02 * (2 ** max(0, rank - 2))
            score += passed_bonus
            # Extra if White king can support the pawn advance
            pawn_file = chess.square_file(sq)
            support_dist = max(abs(wk_rank - rank), abs(wk_file - pawn_file))
            if support_dist <= 2:
                score += passed_bonus * 0.5

    # White queens (promoted pawns)
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    score += white_queens * 0.30

    # ---- King tropism (White king as attacker) ----
    # Chebyshev distance — with double moves, king threatens at distance 2
    king_dist = max(abs(wk_rank - bk_rank), abs(wk_file - bk_file))
    if king_dist <= 2:
        score += 0.15  # imminent capture threat
    elif king_dist <= 4:
        score += 0.05  # within 2-turn striking range

    # Tropism to undefended Black pieces (free captures with double move)
    piece_values = {chess.PAWN: 0.03, chess.KNIGHT: 0.08, chess.BISHOP: 0.08,
                    chess.ROOK: 0.12, chess.QUEEN: 0.20}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == chess.BLACK and piece.piece_type != chess.KING:
            dist = max(abs(wk_rank - chess.square_rank(sq)),
                       abs(wk_file - chess.square_file(sq)))
            if dist <= 2 and not board.is_attacked_by(chess.BLACK, sq):
                score += piece_values.get(piece.piece_type, 0.05)

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
    if black_heavy >= 2:
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
        score -= adjacent_attacked * 0.06

        # Bonus for heavy pieces controlling same rank/file as king
        for sq in board.pieces(chess.QUEEN, chess.BLACK) | board.pieces(chess.ROOK, chess.BLACK):
            sq_rank = chess.square_rank(sq)
            sq_file = chess.square_file(sq)
            if sq_rank == wk_rank or sq_file == wk_file:
                score -= 0.08

    # ---- Material balance (general) ----
    black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    score -= (black_knights + black_bishops) * 0.03

    return max(-0.95, min(0.95, score))


class NNEvaluator:
    """Neural network evaluator wrapping a PyTorch dual-head model.

    Returns value (scalar) from __call__ for backward compatibility
    with heuristic-mode MCTS.  The policy output is available via
    evaluate_with_policy() and batch_evaluate_with_policy().
    """

    def __init__(self, model_path):
        import torch
        from train import DualHeadNet
        from data_processor import fen_to_tensor

        self.torch = torch
        self.fen_to_tensor = fen_to_tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DualHeadNet()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        # FP16 for faster inference on GPU
        if self.device.type == "cuda":
            self.model.half()
            self._half = True
        else:
            self._half = False

    def __call__(self, game_state):
        """Return value only (backward-compatible with heuristic eval)."""
        val, _ = self.evaluate_with_policy(game_state)
        return val

    def evaluate_with_policy(self, game_state):
        """Return (value, policy_logits) for a single game state.

        policy_logits is a numpy array of shape (4096,), or None for
        terminal states.
        """
        board = game_state.board
        if board.king(chess.WHITE) is None:
            return -1.0, None
        if board.king(chess.BLACK) is None:
            return 1.0, None

        tensor = self.fen_to_tensor(
            game_state.fen(),
            is_white_turn=game_state.is_white_turn,
        )
        # (8, 8, C) -> (1, C, 8, 8) channels-first
        inp = self.torch.from_numpy(
            tensor.transpose(2, 0, 1)[np.newaxis]
        ).to(self.device)
        if self._half:
            inp = inp.half()

        with self.torch.no_grad():
            value_out, policy_out = self.model(inp)
        return float(value_out[0, 0].item()), policy_out[0].cpu().float().numpy()

    def batch_evaluate(self, game_states):
        """Evaluate multiple states, returning values only (list of float)."""
        vals, _ = self._batch_impl(game_states)
        return vals

    def batch_evaluate_with_policy(self, game_states):
        """Evaluate multiple states, return (values, policies) lists."""
        return self._batch_impl(game_states)

    def _batch_impl(self, game_states):
        values = []
        policy_list = []
        indices_to_predict = []
        tensors = []

        for i, gs in enumerate(game_states):
            board = gs.board
            if board.king(chess.WHITE) is None:
                values.append(-1.0)
                policy_list.append(None)
            elif board.king(chess.BLACK) is None:
                values.append(1.0)
                policy_list.append(None)
            else:
                values.append(None)
                policy_list.append(None)
                indices_to_predict.append(i)
                tensors.append(self.fen_to_tensor(
                    gs.fen(), is_white_turn=gs.is_white_turn,
                ))

        if tensors:
            # Stack and transpose to channels-first: (N, 8, 8, C) -> (N, C, 8, 8)
            batch_np = np.stack(tensors, axis=0).transpose(0, 3, 1, 2)
            batch = self.torch.from_numpy(batch_np).to(self.device)
            if self._half:
                batch = batch.half()

            with self.torch.no_grad():
                val_out, pol_out = self.model(batch)

            val_np = val_out.cpu().float().numpy().flatten()
            pol_np = pol_out.cpu().float().numpy()
            for j, idx in enumerate(indices_to_predict):
                values[idx] = float(val_np[j])
                policy_list[idx] = pol_np[j]

        return values, policy_list
