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
            self._result = 0
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
        """Generate all legal moves for Black.

        Uses legal_moves so Black can't leave its own king in self-check.
        No additional safety filters — MCTS handles threat awareness.
        """
        self.board.turn = chess.BLACK
        return list(self.board.legal_moves)

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
                    self.is_white_turn = False
                    self.turn_count += 1
                    self._terminal = False
                    self._result = None
                    return True

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
        self.board.turn = chess.BLACK
        legal = list(self.board.legal_moves)
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
