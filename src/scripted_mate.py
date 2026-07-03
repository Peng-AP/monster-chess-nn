"""Deterministic Black conversion for K + heavies (Q/R) vs bare White king in
Monster Chess.

Architecture: a 2-ply minimax — every legal Black move against every White
double-move reply (White has only a king, so <= ~64 replies) — scored by a
static evaluation that encodes the winning plan.  Earlier versions used
hand-written priority rules; against an adversarial searcher they kept
developing rule-interaction bugs (rescue starving king safety, defender
lapses, oscillating repositioning).  The search resolves those tactical
interactions automatically; the evaluation only has to know what a good
position looks like:

- Double fence: one covered rank is permeable (the double-move king passes
  THROUGH an attacked square and lands beyond it), two adjacent covered ranks
  are not.  Reward an adjacent covered pair (a, a+1) above the king, tighter
  is better.
- a <= 5: the fence stops LANDING beyond it, not CAPTURING beyond it; keeping
  the White king's ceiling at vrank 4 leaves the Black king zone (vrank 7)
  out of double-move capture range.
- Immunity: White may capture the Black king only if White ends its turn
  safe, so a DEFENDED Black king is immune (barring forced-blunder cases).
- Mate: shrink the confinement zone; when the zone is fully covered every
  White turn ends attacked and the king is captured next move.

Push direction (which edge White is driven toward) is chosen once, away from
the Black king's half.
"""
import chess

HEAVY_TYPES = (chess.QUEEN, chess.ROOK)
INF = 10 ** 9


def _cheb(a, b):
    return max(abs(chess.square_rank(a) - chess.square_rank(b)),
               abs(chess.square_file(a) - chess.square_file(b)))


class ScriptedMate:
    def __init__(self):
        self.edge = None  # rank (0 or 7) White is pushed toward
        self._visits = {}  # position fen -> times reached by our own moves

    # ------------------------------------------------------------------
    def select_move(self, game):
        board = game.board
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        legal = list(game.get_search_actions())
        if not legal:
            return None
        if wk is None:
            return legal[0]

        # Immediate win.
        for m in legal:
            if m.to_square == wk:
                return m

        if self.edge is None:
            bk_rank = chess.square_rank(bk) if bk is not None else 7
            self.edge = 0 if bk_rank >= 4 else 7

        best_move, best_score, best_key = None, -INF, None
        for m in legal:
            g = game.clone()
            g.apply_search_action(m)
            score = self._white_reply_min(g.board)
            # Anti-repetition: on a score plateau, drifting to novel positions
            # beats shuffling between two of them until the turn cap.
            key = g.board.board_fen()
            score -= 25.0 * self._visits.get(key, 0)
            if score > best_score:
                best_score, best_move, best_key = score, m, key
        if best_key is not None:
            self._visits[best_key] = self._visits.get(best_key, 0) + 1
        return best_move if best_move is not None else legal[0]

    # ------------------------------------------------------------------
    def _white_reply_min(self, board):
        """Worst-case (for Black) score over all White double-move replies."""
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if bk is None:
            return -INF
        if wk is None:
            return INF

        saved_turn = board.turn
        board.turn = chess.WHITE
        worst = INF
        m1s = [m for m in board.pseudo_legal_moves
               if m.from_square == wk]
        m1s.append(None)  # "no improving first step" isn't legal, but a
        # stationary bound costs little and guards the empty-move edge case.
        for m1 in m1s:
            if m1 is not None:
                if m1.to_square == bk:
                    # Capture on the first half wins for White immediately
                    # (rule: only offered/kept when White ends safe, but be
                    # conservative and treat it as a loss unless defended).
                    if not board.is_attacked_by(chess.BLACK, bk):
                        board.turn = saved_turn
                        return -INF
                    continue
                board.push(m1)
                board.turn = chess.WHITE
                wk1 = board.king(chess.WHITE)
            else:
                wk1 = wk
            m2s = [m for m in board.pseudo_legal_moves
                   if m.from_square == wk1] if m1 is not None else []
            if m1 is None:
                s = self._score(board)
                worst = min(worst, s)
            else:
                any_m2 = False
                for m2 in m2s:
                    if m2.to_square == bk:
                        board.push(m2)
                        wk2 = board.king(chess.WHITE)
                        safe = (wk2 is None or
                                not board.is_attacked_by(chess.BLACK, wk2))
                        board.pop()
                        board.turn = chess.WHITE
                        if safe:
                            board.pop()
                            board.turn = saved_turn
                            return -INF  # king falls: position is lost
                        continue
                    any_m2 = True
                    board.push(m2)
                    s = self._score(board)
                    board.pop()
                    board.turn = chess.WHITE
                    worst = min(worst, s)
                if not any_m2:
                    worst = min(worst, self._score(board))
                board.pop()
            if worst == -INF:
                break
        board.turn = saved_turn
        return worst

    # ------------------------------------------------------------------
    def _score(self, board):
        """Static evaluation after White's reply (Black to move). Higher =
        better for Black."""
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if bk is None:
            return -INF
        if wk is None:
            return INF

        vr = (lambda r: r) if self.edge == 0 else (lambda r: 7 - r)
        vwk = vr(chess.square_rank(wk))
        score = 0.0

        heavies = [sq for sq in chess.SQUARES
                   if (p := board.piece_at(sq)) is not None
                   and p.color == chess.BLACK and p.piece_type in HEAVY_TYPES]

        # Black can capture the White king right now -> winning next ply.
        if board.is_attacked_by(chess.BLACK, wk):
            score += 900.0

        # King safety / immunity.
        bk_defended = board.is_attacked_by(chess.BLACK, bk)
        d_kings = _cheb(bk, wk)
        if d_kings <= 2 and not bk_defended:
            score -= 5000.0  # White captures next turn
        if d_kings <= 4:
            # Defense makes proximity survivable, not desirable — a big bonus
            # here taught the king to camp beside White, blocking fence ranks.
            score += 4.0 if bk_defended else -30.0
        score += min(d_kings, 5) * 2.0

        # Material: a heavy within double-move range is (conservatively) lost.
        score += 40.0 * len(heavies)
        for sq in heavies:
            if _cheb(sq, wk) <= 2:
                score -= 35.0

        # Rank coverage (a Black piece on the rank whose line isn't blocked
        # from covering; own king mid-file on the rank spoils the cut).
        bk_rank = chess.square_rank(bk)
        bk_mid = chess.square_file(bk) not in (0, 7)
        covered = set()
        for sq in heavies:
            r = chess.square_rank(sq)
            if r == bk_rank and bk_mid:
                continue
            covered.add(vr(r))

        # Fence: tightest adjacent covered pair strictly above the king,
        # capped at 5 (capture-through-the-fence bound).
        fence = None
        for a in sorted(covered):
            if vwk < a <= 5 and (a + 1) in covered:
                fence = a
                break
        if fence is not None:
            score += 120.0 + (5 - fence) * 25.0
            if fence - 1 in covered:
                score += 45.0  # squeeze rank in place
                if fence - 1 == 0 or (fence - 1) == vwk:
                    score += 20.0
            else:
                # Rotation shaping: a 2-ply search cannot bridge "move to a
                # clear file now, descend to the squeeze rank next turn"
                # without a gradient on the intermediate step.  Reward a free
                # heavy standing on a file from which the squeeze rank
                # (fence-1) is reachable in one safe move.
                real_sq = (lambda f, r: chess.square(f, r)) if self.edge == 0 \
                    else (lambda f, r: chess.square(f, 7 - r))
                wkf = chess.square_file(wk)
                fence_sqs = set()
                for r in (fence, fence + 1):
                    for sq in heavies:
                        if vr(chess.square_rank(sq)) == r:
                            fence_sqs.add(sq)
                            break
                for sq in heavies:
                    if sq in fence_sqs:
                        continue
                    f = chess.square_file(sq)
                    if abs(f - wkf) < 3:
                        continue
                    r_cur = chess.square_rank(sq)
                    r_tgt = chess.square_rank(real_sq(f, fence - 1))
                    step = 1 if r_tgt > r_cur else -1
                    if r_cur == r_tgt:
                        continue
                    clear = all(board.piece_at(chess.square(f, r)) is None
                                for r in range(r_cur + step, r_tgt + step, step))
                    landing = chess.square(f, r_tgt)
                    if clear and _cheb(landing, wk) >= 3:
                        score += 12.0
                        break
        else:
            # Build gradient: reward covered ranks in the build zone and
            # heavies close to it.
            base = 5 if vwk < 5 else None
            if base is not None:
                for r in (base, base + 1):
                    if r in covered:
                        score += 25.0
                near = sorted(abs(vr(chess.square_rank(sq)) - base)
                              for sq in heavies)
                for gap in near[:2]:
                    score -= gap * 1.5
            score -= 15.0  # no fence is always worse than a fence

        # White king attacked with nowhere unattacked to land is mate-like;
        # approximated by rewarding a small, increasingly covered zone.
        if fence is not None:
            score += (5 - fence) * 2.0
            score -= vwk * 3.0  # push him toward the edge

        # Keep heavies at safe working distance (cheap shaping).
        for sq in heavies:
            d = _cheb(sq, wk)
            if d == 3:
                score += 1.0

        # Keep the Black king drifting toward its home zone (vrank 7) so it
        # neither wanders into the push zone nor blocks fence ranks mid-file.
        score += min(vr(bk_rank), 6) * 1.5

        # A mid-file king standing on a working rank spoils the cut there.
        if bk_mid:
            working = ({fence - 1, fence, fence + 1} if fence is not None
                       else {5, 6})
            if vr(bk_rank) in working:
                score -= 22.0
        return score
