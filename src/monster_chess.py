import chess
import copy
import random
from config import STARTING_FEN, MAX_GAME_TURNS
from evaluation import evaluate as _heuristic_evaluate


class MonsterChessGame:
    """Wraps python-chess to enforce Monster Chess rules.

    Monster Chess:
      - White has King + 4 pawns, gets TWO consecutive moves per turn.
      - Black has the full army, gets ONE move per turn.
      - A king capture ends the game (no check/checkmate — captures are legal).
      - No castling for White (only king on back rank).
    """

    def __init__(self, fen=STARTING_FEN):
        self.board = chess.Board(fen)
        self.is_white_turn = self.board.turn == chess.WHITE
        self.turn_count = 0
        self._terminal = False
        self._result = None  # +1 white wins, -1 black wins, 0 draw

    def clone(self):
        g = MonsterChessGame.__new__(MonsterChessGame)
        g.board = self.board.copy()
        g.is_white_turn = self.is_white_turn
        g.turn_count = self.turn_count
        g._terminal = self._terminal
        g._result = self._result
        return g

    def current_player(self):
        return chess.WHITE if self.is_white_turn else chess.BLACK

    def fen(self):
        return self.board.fen()

    # ------------------------------------------------------------------
    # Terminal / result
    # ------------------------------------------------------------------

    def is_terminal(self):
        if self._terminal:
            return True
        if self.board.king(chess.WHITE) is None:
            self._terminal = True
            self._result = -1
        elif self.board.king(chess.BLACK) is None:
            self._terminal = True
            self._result = 1
        elif self.turn_count >= MAX_GAME_TURNS:
            self._terminal = True
            # Use heuristic to re-label move-limit draws: if Black has a
            # dominant material advantage (score < -0.4), record -0.5 instead
            # of 0 so the model learns these are Black-favored, not neutral.
            h = _heuristic_evaluate(self)
            self._result = -0.5 if h < -0.4 else 0
        # Skip the expensive get_legal_actions() check — stalemate is
        # extremely rare in Monster Chess and not worth the O(n²) cost
        # on every terminal check during MCTS.
        return self._terminal

    def get_result(self):
        """Return result from White's perspective: +1, -1, or 0."""
        if not self._terminal:
            self.is_terminal()
        return self._result if self._result is not None else 0

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def _white_single_moves(self):
        """All pseudo-legal moves for White on the current board."""
        self.board.turn = chess.WHITE
        return list(self.board.pseudo_legal_moves)

    def get_legal_actions(self):
        """Return legal actions for the current player.

        White actions are (move1, move2) tuples.
        Black actions are single Move objects.
        """
        if self._terminal:
            return []

        if self.is_white_turn:
            return self._get_white_actions()
        else:
            return self._get_black_actions()

    def _get_white_actions(self):
        """Generate all legal double-move pairs for White.

        White CAN put Black's king in check (threaten capture).
        White CANNOT leave its own king attacked (self-preservation).
        Exception: if ALL moves leave White's king attacked (checkmate),
        return all moves — White is forced to blunder and Black captures.
        """
        first_moves = self._white_single_moves()
        if not first_moves:
            return []

        safe_pairs = []
        all_pairs = []
        for m1 in first_moves:
            self.board.push(m1)

            # If first move captured Black's king, game over — pair with null
            if self.board.king(chess.BLACK) is None:
                white_king = self.board.king(chess.WHITE)
                white_attacked = (
                    white_king is not None
                    and self.board.is_attacked_by(chess.BLACK, white_king)
                )
                self.board.pop()
                if not white_attacked:
                    return [(m1, chess.Move.null())]
                # Unsafe king-capture is illegal under self-preservation.
                continue

            self.board.turn = chess.WHITE
            second_moves = list(self.board.pseudo_legal_moves)

            for m2 in second_moves:
                self.board.push(m2)

                # If second move captured Black's king, instant win
                if self.board.king(chess.BLACK) is None:
                    white_king = self.board.king(chess.WHITE)
                    white_attacked = (
                        white_king is not None
                        and self.board.is_attacked_by(chess.BLACK, white_king)
                    )
                    self.board.pop()
                    if not white_attacked:
                        self.board.turn = chess.BLACK
                        self.board.pop()
                        return [(m1, m2)]
                    # Unsafe king-capture is illegal under self-preservation.
                    continue

                all_pairs.append((m1, m2))

                # White king must not be left attackable
                white_king = self.board.king(chess.WHITE)
                white_attacked = (
                    white_king is not None
                    and self.board.is_attacked_by(chess.BLACK, white_king)
                )
                if not white_attacked:
                    safe_pairs.append((m1, m2))

                self.board.pop()

            self.board.turn = chess.BLACK
            self.board.pop()

        # If no safe moves, White is forced to blunder (Black captures)
        return safe_pairs if safe_pairs else all_pairs

    def _get_black_actions(self):
        """Generate all legal single moves for Black.

        Monster Chess ends on king capture, so Black must be able to capture
        White's king directly (python-chess legal_moves forbids this).
        We therefore use pseudo-legal moves and apply the same
        self-preservation policy used for White:
          - keep moves that do not leave Black king attacked by White
          - if no safe moves exist, allow all moves (forced blunder)
        """
        self.board.turn = chess.BLACK
        candidate_moves = list(self.board.pseudo_legal_moves)
        if not candidate_moves:
            return []

        safe_moves = []
        all_moves = []
        for move in candidate_moves:
            self.board.push(move)

            # Immediate king capture wins if Black king remains safe.
            if self.board.king(chess.WHITE) is None:
                black_king = self.board.king(chess.BLACK)
                black_attacked = (
                    black_king is not None
                    and self.board.is_attacked_by(chess.WHITE, black_king)
                )
                self.board.pop()
                if not black_attacked:
                    return [move]
                continue

            all_moves.append(move)
            black_king = self.board.king(chess.BLACK)
            black_attacked = (
                black_king is not None
                and self.board.is_attacked_by(chess.WHITE, black_king)
            )
            if not black_attacked:
                safe_moves.append(move)
            self.board.pop()

        return safe_moves if safe_moves else all_moves

    # ------------------------------------------------------------------
    # Applying actions
    # ------------------------------------------------------------------

    def apply_action(self, action):
        """Apply an action in-place and advance the game state."""
        if self.is_white_turn:
            m1, m2 = action
            self.board.push(m1)
            if self.board.king(chess.BLACK) is not None and m2 != chess.Move.null():
                self.board.turn = chess.WHITE
                # Verify m2 is still pseudo-legal after m1 changed the board
                if m2 in self.board.pseudo_legal_moves:
                    self.board.push(m2)
        else:
            self.board.push(action)

        self.is_white_turn = not self.is_white_turn
        self.turn_count += 1
        self._terminal = False
        self._result = None

    def apply_random_action(self):
        """Pick and apply a random legal action. Much faster than
        get_legal_actions() for White because it samples single moves
        instead of enumerating all pairs.

        Returns True if a move was made, False if no moves available.
        """
        if self._terminal:
            return False

        if self.is_white_turn:
            return self._apply_random_white()
        else:
            return self._apply_random_black()

    def _apply_random_white(self):
        """Sample a random safe double-move for White without enumerating all pairs."""
        self.board.turn = chess.WHITE
        first_moves = list(self.board.pseudo_legal_moves)
        if not first_moves:
            return False

        random.shuffle(first_moves)
        for m1 in first_moves:
            self.board.push(m1)
            if self.board.king(chess.BLACK) is None:
                white_king = self.board.king(chess.WHITE)
                white_attacked = (
                    white_king is not None
                    and self.board.is_attacked_by(chess.BLACK, white_king)
                )
                if not white_attacked:
                    self.is_white_turn = False
                    self.turn_count += 1
                    self._terminal = False
                    self._result = None
                    return True
                self.board.pop()
                continue

            self.board.turn = chess.WHITE
            second_moves = list(self.board.pseudo_legal_moves)
            random.shuffle(second_moves)

            for m2 in second_moves:
                self.board.push(m2)
                if self.board.king(chess.BLACK) is None:
                    white_king = self.board.king(chess.WHITE)
                    white_safe = (
                        white_king is None
                        or not self.board.is_attacked_by(chess.BLACK, white_king)
                    )
                    if white_safe:
                        self.is_white_turn = False
                        self.turn_count += 1
                        self._terminal = False
                        self._result = None
                        return True
                    self.board.pop()
                    continue

                white_king = self.board.king(chess.WHITE)
                white_safe = (
                    white_king is None
                    or not self.board.is_attacked_by(chess.BLACK, white_king)
                )
                if white_safe:
                    self.is_white_turn = False
                    self.turn_count += 1
                    self._terminal = False
                    self._result = None
                    return True

                self.board.pop()

            self.board.turn = chess.BLACK
            self.board.pop()

        # No safe moves — pick any move (forced blunder)
        m1 = random.choice(first_moves)
        self.board.push(m1)
        self.board.turn = chess.WHITE
        second_moves = list(self.board.pseudo_legal_moves)
        if second_moves:
            m2 = random.choice(second_moves)
            self.board.push(m2)
        self.is_white_turn = False
        self.turn_count += 1
        self._terminal = False
        self._result = None
        return True

    def _apply_random_black(self):
        """Sample a random legal move for Black."""
        legal = self._get_black_actions()
        if not legal:
            return False

        move = random.choice(legal)
        self.board.push(move)
        self.is_white_turn = True
        self.turn_count += 1
        self._terminal = False
        self._result = None
        return True

    # ------------------------------------------------------------------
    # Utility for action encoding (used by data generation)
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_str(action, is_white):
        if is_white:
            m1, m2 = action
            return f"{m1.uci()},{m2.uci()}"
        else:
            return action.uci()

    @staticmethod
    def str_to_action(s, is_white):
        if is_white:
            parts = s.split(",")
            return (chess.Move.from_uci(parts[0]), chess.Move.from_uci(parts[1]))
        else:
            return chess.Move.from_uci(s)
