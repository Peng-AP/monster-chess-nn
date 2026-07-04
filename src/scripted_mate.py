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
- King capture wins UNCONDITIONALLY (double-move-check rule, confirmed
  2026-07-04): there is no "defended king immunity".  The Black king's only
  safety is distance — Chebyshev >= 3 from the White king whenever White is
  to move.
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
            score = self._white_reply_min(g.board, depth=2)
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
    def _white_reply_min(self, board, depth=0):
        """Worst-case (for Black) score over all White double-move replies.

        depth > 0 enables a danger-gated extension: when a reply leaves the
        Black king in the chase zone (Chebyshev <= 3), the leaf is resolved by
        a king-escape search one Black+White level deeper instead of the
        static score — 2-ply alone cannot see multi-turn king hunts, which
        was the residual ~6% death rate.
        """
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
                    # Capture on the first half wins for White immediately —
                    # unconditionally (double-move-check rule).
                    board.turn = saved_turn
                    return -INF
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
                        # Unconditional loss: king capture wins regardless of
                        # the capturer's own safety.
                        board.pop()
                        board.turn = saved_turn
                        return -INF
                    any_m2 = True
                    board.push(m2)
                    s = self._leaf(board, depth)
                    board.pop()
                    board.turn = chess.WHITE
                    worst = min(worst, s)
                if not any_m2:
                    worst = min(worst, self._leaf(board, depth))
                board.pop()
            if worst == -INF:
                break
        board.turn = saved_turn
        return worst

    def _leaf(self, board, depth):
        """Leaf resolution: static score, or a king-escape extension when the
        Black king is in the chase zone and depth remains."""
        if depth <= 0:
            return self._score(board)
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is None or bk is None or _cheb(bk, wk) > 3:
            return self._score(board)
        return self._black_escape_max(board, depth - 1)

    def _black_escape_max(self, board, depth):
        """Black-to-move escape search: king steps plus win-by-capture, each
        answered by White's minimizing reply.  Restricted to king moves — the
        point is resolving the chase, not full-width search."""
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is None:
            return INF
        if bk is None:
            return -INF
        saved_turn = board.turn
        board.turn = chess.BLACK
        best = None
        for m in list(board.pseudo_legal_moves):
            if m.to_square == wk:
                board.turn = saved_turn
                return INF  # capture the White king: win
            if m.from_square != bk:
                continue
            board.push(m)
            s = self._white_reply_min(board, depth=depth)
            board.pop()
            board.turn = chess.BLACK
            best = s if best is None else max(best, s)
        board.turn = saved_turn
        # No king move at all: fall back to the static score.
        return best if best is not None else self._score(board)

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

        # King safety is DISTANCE ONLY (no defended-king immunity under the
        # corrected rules: king capture wins unconditionally).
        d_kings = _cheb(bk, wk)
        if d_kings <= 2:
            score -= 5000.0  # White captures next turn
        elif d_kings <= 4:
            # In the chase zone the king lives or dies by ESCAPE GEOMETRY —
            # a cornered king with no square that restores distance >= 3 is
            # dead in a few turns even though no single 2-ply line shows it.
            escapes = 0
            bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0:
                        continue
                    f, r = bkf + df, bkr + dr
                    if 0 <= f <= 7 and 0 <= r <= 7:
                        to = chess.square(f, r)
                        if board.piece_at(to) is None and _cheb(to, wk) >= 3:
                            escapes += 1
            if escapes == 0:
                score -= 4000.0
            elif escapes == 1:
                score -= 120.0
            else:
                score -= 40.0 if d_kings == 3 else 10.0
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
            # A cover the king can capture this turn (Chebyshev <= 2, transit
            # through attacked squares is legal) is already gone — counting it
            # made the search hold a fence while its rook was being eaten.
            if _cheb(sq, wk) < 3:
                continue
            covered.add(vr(r))

        # Fence: tightest adjacent covered pair strictly above the king,
        # capped at 5 (capture-through-the-fence bound).
        fence = None
        for a in sorted(covered):
            if vwk < a <= 5 and (a + 1) in covered:
                fence = a
                break
        if vwk >= 5:
            # White king inside the Black zone: crisis under corrected rules
            # (no immunity — only fences keep the Black king alive).
            score -= 150.0

        if fence is not None:
            # The fence is EVERYTHING: without immunity, a lone king is
            # always run down in the open (White gains one Chebyshev step per
            # turn).  These weights must dominate all shaping terms so the
            # 2-ply search spends its first tempi completing the fence.
            score += 200.0 + (5 - fence) * 25.0
            bk_vr = vr(bk_rank)
            # Safe side of the fence: White's ceiling is fence-1 (landing),
            # capture reach fence+1, so the king needs vrank >= fence+2.
            score += 60.0 if bk_vr >= fence + 2 else -60.0
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
                        score += 60.0
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
            # Heavies hugging their own king block its escape squares — the
            # clustered-corner starts died exactly this way.  Untangle early.
            if _cheb(sq, bk) <= 1:
                score -= 10.0

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
