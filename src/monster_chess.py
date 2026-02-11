import chess
import copy
import random
from config import STARTING_FEN, MAX_GAME_TURNS


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
        self.is_white_turn = True
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
            self._result = 0
        elif not self.get_legal_actions():
            self._terminal = True
            self._result = 0
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
        """Generate all legal double-move pairs for White."""
        first_moves = self._white_single_moves()
        if not first_moves:
            return []

        pairs = []
        for m1 in first_moves:
            self.board.push(m1)

            # If first move captured Black's king, game over — pair with null
            if self.board.king(chess.BLACK) is None:
                self.board.pop()
                return [(m1, chess.Move.null())]

            self.board.turn = chess.WHITE
            second_moves = list(self.board.pseudo_legal_moves)

            for m2 in second_moves:
                self.board.push(m2)

                # If second move captured Black's king, instant win
                if self.board.king(chess.BLACK) is None:
                    self.board.pop()
                    self.board.turn = chess.BLACK
                    self.board.pop()
                    return [(m1, m2)]

                # After White's double-move the position must be legal:
                # 1. Black's king must not be in check (board.turn is Black here)
                # 2. White's king must not be attackable (can't leave king hanging)
                black_in_check = self.board.is_check()
                white_king = self.board.king(chess.WHITE)
                white_attacked = (
                    white_king is not None
                    and self.board.is_attacked_by(chess.BLACK, white_king)
                )
                if not black_in_check and not white_attacked:
                    pairs.append((m1, m2))

                self.board.pop()

            self.board.turn = chess.BLACK
            self.board.pop()

        return pairs

    def _get_black_actions(self):
        """Generate all legal moves for Black.

        Uses legal_moves (not pseudo_legal) so Black cannot leave its
        own king in check.  Also filters out moves that leave the king
        capturable by White's first move next turn — Black must "see"
        immediate one-move threats from White.
        """
        self.board.turn = chess.BLACK
        safe_moves = []
        for move in self.board.legal_moves:
            self.board.push(move)
            # Check if White can capture Black's king with a single move
            self.board.turn = chess.WHITE
            king_sq = self.board.king(chess.BLACK)
            if king_sq is not None and self.board.is_attacked_by(chess.WHITE, king_sq):
                # White can reach the king in one move — unsafe
                self.board.pop()
                continue
            self.board.pop()
            safe_moves.append(move)
        # If every move leaves the king capturable, allow all legal moves
        # (better to have options than be stuck)
        if not safe_moves:
            self.board.turn = chess.BLACK
            return list(self.board.legal_moves)
        return safe_moves

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
        """Sample a random legal double-move for White without enumerating all pairs."""
        self.board.turn = chess.WHITE
        first_moves = list(self.board.pseudo_legal_moves)
        if not first_moves:
            return False

        random.shuffle(first_moves)
        for m1 in first_moves:
            self.board.push(m1)
            if self.board.king(chess.BLACK) is None:
                # Captured Black's king on first move
                self.is_white_turn = False
                self.turn_count += 1
                self._terminal = False
                self._result = None
                return True

            self.board.turn = chess.WHITE
            second_moves = list(self.board.pseudo_legal_moves)
            random.shuffle(second_moves)

            for m2 in second_moves:
                self.board.push(m2)
                if self.board.king(chess.BLACK) is None:
                    # Captured Black's king on second move — keep it
                    self.is_white_turn = False
                    self.turn_count += 1
                    self._terminal = False
                    self._result = None
                    return True

                black_in_check = self.board.is_check()
                white_king = self.board.king(chess.WHITE)
                white_attacked = (
                    white_king is not None
                    and self.board.is_attacked_by(chess.BLACK, white_king)
                )
                if not black_in_check and not white_attacked:
                    # Found a valid pair — keep the board state
                    self.is_white_turn = False
                    self.turn_count += 1
                    self._terminal = False
                    self._result = None
                    return True

                self.board.pop()

            self.board.turn = chess.BLACK
            self.board.pop()

        return False

    def _apply_random_black(self):
        """Sample a random legal move for Black, preferring moves that
        don't leave the king capturable by White in one move."""
        self.board.turn = chess.BLACK
        legal = list(self.board.legal_moves)
        if not legal:
            return False

        # Try to find a move that doesn't leave king attacked
        random.shuffle(legal)
        for move in legal:
            self.board.push(move)
            self.board.turn = chess.WHITE
            king_sq = self.board.king(chess.BLACK)
            if king_sq is None or not self.board.is_attacked_by(chess.WHITE, king_sq):
                # Safe — keep this move
                self.is_white_turn = True
                self.turn_count += 1
                self._terminal = False
                self._result = None
                return True
            self.board.pop()

        # All moves leave king capturable — just pick one
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
