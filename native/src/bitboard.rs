//! Board state, FEN, attack detection and pseudo-legal move generation.
//!
//! The reference is **python-chess as the Python engine calls it**, not the
//! rules of chess. Two consequences that shape this file:
//!
//! 1. **Castling is check-aware here even though it is called "pseudo-legal".**
//!    python-chess refuses castling out of, through, or into check in its
//!    pseudo-legal generator, so Monster Chess's White — which may otherwise
//!    step onto an attacked square — cannot castle through one. Verified
//!    against the live engine 2026-08-03 and pinned in
//!    `tests/test_ruleset_divergences.py`. A from-scratch generator would
//!    naturally produce the other answer.
//! 2. **King captures are ordinary pseudo-legal moves.** Nothing here treats a
//!    king as un-capturable; that is the whole variant.
//!
//! Square indexing matches python-chess: a1 = 0, h8 = 63, rank = sq >> 3.

use std::sync::LazyLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub const BLACK: usize = 0;
pub const WHITE: usize = 1;

pub const PAWN: u8 = 1;
pub const KNIGHT: u8 = 2;
pub const BISHOP: u8 = 3;
pub const ROOK: u8 = 4;
pub const QUEEN: u8 = 5;
pub const KING: u8 = 6;

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;

#[inline]
fn rank_of(sq: u8) -> u8 {
    sq >> 3
}

#[inline]
fn file_of(sq: u8) -> u8 {
    sq & 7
}

// ---------------------------------------------------------------------------
// Precomputed leaper attacks
// ---------------------------------------------------------------------------

static KNIGHT_ATTACKS: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut t = [0u64; 64];
    for sq in 0..64usize {
        let b = 1u64 << sq;
        let mut a = 0u64;
        a |= (b << 17) & !FILE_A;
        a |= (b << 15) & !FILE_H;
        a |= (b << 10) & !(FILE_A | (FILE_A << 1));
        a |= (b << 6) & !(FILE_H | (FILE_H >> 1));
        a |= (b >> 17) & !FILE_H;
        a |= (b >> 15) & !FILE_A;
        a |= (b >> 10) & !(FILE_H | (FILE_H >> 1));
        a |= (b >> 6) & !(FILE_A | (FILE_A << 1));
        t[sq] = a;
    }
    t
});

static KING_ATTACKS: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut t = [0u64; 64];
    for sq in 0..64usize {
        let b = 1u64 << sq;
        let mut a = 0u64;
        a |= (b << 8) | (b >> 8);
        a |= (b << 1) & !FILE_A;
        a |= (b << 9) & !FILE_A;
        a |= (b >> 7) & !FILE_A;
        a |= (b >> 1) & !FILE_H;
        a |= (b >> 9) & !FILE_H;
        a |= (b << 7) & !FILE_H;
        t[sq] = a;
    }
    t
});

/// PAWN_ATTACKS[color][sq] — squares a pawn of `color` on `sq` attacks.
static PAWN_ATTACKS: LazyLock<[[u64; 64]; 2]> = LazyLock::new(|| {
    let mut t = [[0u64; 64]; 2];
    for sq in 0..64usize {
        let b = 1u64 << sq;
        t[WHITE][sq] = ((b << 9) & !FILE_A) | ((b << 7) & !FILE_H);
        t[BLACK][sq] = ((b >> 7) & !FILE_A) | ((b >> 9) & !FILE_H);
    }
    t
});

const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Sliding attacks by ray-walking. Deliberately simple: E1 is a parity phase,
/// and magics are an optimisation to be added later behind the same tests.
fn sliding_attacks(sq: u8, occupied: u64, dirs: &[(i8, i8); 4]) -> u64 {
    let mut attacks = 0u64;
    let (f0, r0) = (file_of(sq) as i8, rank_of(sq) as i8);
    for &(df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let target = (r * 8 + f) as u8;
            attacks |= 1u64 << target;
            if occupied & (1u64 << target) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    attacks
}

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Board {
    pub pawns: u64,
    pub knights: u64,
    pub bishops: u64,
    pub rooks: u64,
    pub queens: u64,
    pub kings: u64,
    pub occupied_co: [u64; 2],
    pub occupied: u64,
    pub turn: bool, // true = White
    /// Bitboard of ROOK squares still carrying rights, as python-chess stores it.
    pub castling: u64,
    pub ep_square: Option<u8>,
    pub halfmove: u32,
    pub fullmove: u32,
}

impl Board {
    pub fn empty() -> Self {
        Board {
            pawns: 0,
            knights: 0,
            bishops: 0,
            rooks: 0,
            queens: 0,
            kings: 0,
            occupied_co: [0, 0],
            occupied: 0,
            turn: true,
            castling: 0,
            ep_square: None,
            halfmove: 0,
            fullmove: 1,
        }
    }

    pub fn piece_type_at(&self, sq: u8) -> Option<u8> {
        let b = 1u64 << sq;
        if self.occupied & b == 0 {
            return None;
        }
        Some(if self.pawns & b != 0 {
            PAWN
        } else if self.knights & b != 0 {
            KNIGHT
        } else if self.bishops & b != 0 {
            BISHOP
        } else if self.rooks & b != 0 {
            ROOK
        } else if self.queens & b != 0 {
            QUEEN
        } else {
            KING
        })
    }

    pub fn color_at(&self, sq: u8) -> Option<usize> {
        let b = 1u64 << sq;
        if self.occupied_co[WHITE] & b != 0 {
            Some(WHITE)
        } else if self.occupied_co[BLACK] & b != 0 {
            Some(BLACK)
        } else {
            None
        }
    }

    fn set_piece(&mut self, sq: u8, piece: u8, color: usize) {
        let b = 1u64 << sq;
        match piece {
            PAWN => self.pawns |= b,
            KNIGHT => self.knights |= b,
            BISHOP => self.bishops |= b,
            ROOK => self.rooks |= b,
            QUEEN => self.queens |= b,
            KING => self.kings |= b,
            _ => {}
        }
        self.occupied_co[color] |= b;
        self.occupied |= b;
    }

    pub fn king_square(&self, color: usize) -> Option<u8> {
        let bb = self.kings & self.occupied_co[color];
        if bb == 0 {
            None
        } else {
            Some(bb.trailing_zeros() as u8)
        }
    }

    /// Pieces of `color` that attack `sq`. Used for castling safety and, at the
    /// Monster layer, for "can this side capture the king right now".
    pub fn attackers(&self, color: usize, sq: u8) -> u64 {
        let occ = self.occupied;
        let mut result = 0u64;
        result |= KNIGHT_ATTACKS[sq as usize] & self.knights;
        result |= KING_ATTACKS[sq as usize] & self.kings;
        let rook_like = self.rooks | self.queens;
        let bishop_like = self.bishops | self.queens;
        result |= sliding_attacks(sq, occ, &ROOK_DIRS) & rook_like;
        result |= sliding_attacks(sq, occ, &BISHOP_DIRS) & bishop_like;
        // A pawn of `color` attacks `sq` iff a pawn of the OPPOSITE colour on
        // `sq` would attack that pawn's square.
        let opposite = if color == WHITE { BLACK } else { WHITE };
        result |= PAWN_ATTACKS[opposite][sq as usize] & self.pawns;
        result & self.occupied_co[color]
    }

    pub fn is_attacked_by(&self, color: usize, sq: u8) -> bool {
        self.attackers(color, sq) != 0
    }

    /// Rights that actually count, mirroring python-chess `clean_castling_rights`.
    ///
    /// A side's rights are void unless its king stands on its home square, so a
    /// captured or wandering king silently voids them. In this variant the
    /// Black king really does get captured, which is how this surfaced: the
    /// engine's FEN drops "kq" the moment a White piece lands on e8, and a port
    /// that only cleared rook squares kept them. Caught by the API
    /// differential, 2026-08-03.
    pub fn clean_castling(&self) -> u64 {
        const E1: u64 = 1u64 << 4;
        const E8: u64 = 1u64 << 60;
        const RANK1: u64 = 0xFF;
        const RANK8: u64 = 0xFFu64 << 56;
        let rights = self.castling & self.rooks;
        let mut out = 0u64;
        if self.kings & self.occupied_co[WHITE] & E1 != 0 {
            out |= rights & RANK1 & self.occupied_co[WHITE];
        }
        if self.kings & self.occupied_co[BLACK] & E8 != 0 {
            out |= rights & RANK8 & self.occupied_co[BLACK];
        }
        out
    }

    fn remove_piece(&mut self, sq: u8) {
        let b = !(1u64 << sq);
        self.pawns &= b;
        self.knights &= b;
        self.bishops &= b;
        self.rooks &= b;
        self.queens &= b;
        self.kings &= b;
        self.occupied_co[WHITE] &= b;
        self.occupied_co[BLACK] &= b;
        self.occupied &= b;
    }

    /// Apply a pseudo-legal move, reproducing python-chess's bookkeeping.
    pub fn push(&mut self, mv: &Move) {
        let us = if self.turn { WHITE } else { BLACK };
        let piece = match self.piece_type_at(mv.from) {
            Some(p) => p,
            None => return,
        };
        let is_capture = self.occupied & (1u64 << mv.to) != 0;

        // En passant: the captured pawn is not on the destination square.
        if piece == PAWN && Some(mv.to) == self.ep_square && !is_capture {
            let captured = if self.turn { mv.to - 8 } else { mv.to + 8 };
            self.remove_piece(captured);
        }

        // Castling moves the rook too. Detected the way python-chess does for
        // standard positions: a king travelling two files.
        let castling = piece == KING
            && (file_of(mv.from) as i8 - file_of(mv.to) as i8).abs() == 2;

        self.remove_piece(mv.to);
        self.remove_piece(mv.from);
        let placed = mv.promotion.unwrap_or(piece);
        self.set_piece(mv.to, placed, us);

        if castling {
            let back = rank_of(mv.from) * 8;
            let (rook_from, rook_to) = if file_of(mv.to) == 6 {
                (back + 7, back + 5)
            } else {
                (back, back + 3)
            };
            self.remove_piece(rook_from);
            self.set_piece(rook_to, ROOK, us);
        }

        // Rights die when the rook square is vacated or captured, and all of a
        // side's rights die when its king moves.
        self.castling &= !(1u64 << mv.from) & !(1u64 << mv.to);
        if piece == KING {
            let back_mask = if us == WHITE { 0xFFu64 } else { 0xFFu64 << 56 };
            self.castling &= !back_mask;
        }

        // A double pawn push always sets the square; whether it is *shown* in
        // the FEN is a separate question (see has_legal_ep).
        self.ep_square = if piece == PAWN
            && (rank_of(mv.from) as i8 - rank_of(mv.to) as i8).abs() == 2
        {
            Some(((mv.from as i16 + mv.to as i16) / 2) as u8)
        } else {
            None
        };

        if piece == PAWN || is_capture {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        if !self.turn {
            self.fullmove += 1;
        }
        self.turn = !self.turn;
    }

    /// Does the side to move have a *legal* en-passant capture?
    ///
    /// python-chess's `fen()` defaults to `en_passant="legal"`, so it omits an
    /// ep square that cannot actually be taken. Reproducing that is not
    /// cosmetic: without it every FEN after a double push diverges from the
    /// engine's, and the replay gate compares FENs.
    pub fn has_legal_ep(&self) -> bool {
        let ep = match self.ep_square {
            Some(sq) => sq,
            None => return false,
        };
        if self.occupied & (1u64 << ep) != 0 {
            return false;
        }
        let us = if self.turn { WHITE } else { BLACK };
        let them = if self.turn { BLACK } else { WHITE };
        let rank_mask: u64 = 0xFFu64 << (8 * if self.turn { 4 } else { 3 });
        let mut movers =
            PAWN_ATTACKS[them][ep as usize] & self.pawns & self.occupied_co[us] & rank_mask;
        while movers != 0 {
            let from = movers.trailing_zeros() as u8;
            movers &= movers - 1;
            let mut probe = self.clone();
            probe.push(&Move { from, to: ep, promotion: None });
            // python-chess treats a missing king as safe rather than erroring.
            match probe.king_square(us) {
                None => return true,
                Some(k) => {
                    if !probe.is_attacked_by(them, k) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// FEN
// ---------------------------------------------------------------------------

fn square_from_name(name: &str) -> Option<u8> {
    let bytes = name.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let file = bytes[0].wrapping_sub(b'a');
    let rank = bytes[1].wrapping_sub(b'1');
    if file < 8 && rank < 8 {
        Some(rank * 8 + file)
    } else {
        None
    }
}

/// Public alias so other modules can print squares without duplicating this.
pub fn square_name_pub(sq: u8) -> String {
    square_name(sq)
}

fn square_name(sq: u8) -> String {
    let mut s = String::with_capacity(2);
    s.push((b'a' + file_of(sq)) as char);
    s.push((b'1' + rank_of(sq)) as char);
    s
}

pub fn parse_fen(fen: &str) -> Result<Board, String> {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    if parts.is_empty() {
        return Err("empty FEN".into());
    }
    let mut board = Board::empty();

    let mut rank: i32 = 7;
    let mut file: i32 = 0;
    for ch in parts[0].chars() {
        match ch {
            '/' => {
                rank -= 1;
                file = 0;
            }
            '1'..='8' => file += ch.to_digit(10).unwrap() as i32,
            _ => {
                let color = if ch.is_ascii_uppercase() { WHITE } else { BLACK };
                let piece = match ch.to_ascii_lowercase() {
                    'p' => PAWN,
                    'n' => KNIGHT,
                    'b' => BISHOP,
                    'r' => ROOK,
                    'q' => QUEEN,
                    'k' => KING,
                    other => return Err(format!("bad piece '{other}'")),
                };
                if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                    return Err("FEN piece placement out of range".into());
                }
                board.set_piece((rank * 8 + file) as u8, piece, color);
                file += 1;
            }
        }
    }

    board.turn = parts.get(1).map_or(true, |s| *s != "b");

    board.castling = 0;
    if let Some(rights) = parts.get(2) {
        for ch in rights.chars() {
            match ch {
                'K' => board.castling |= 1u64 << 7,  // h1
                'Q' => board.castling |= 1u64,       // a1
                'k' => board.castling |= 1u64 << 63, // h8
                'q' => board.castling |= 1u64 << 56, // a8
                _ => {}
            }
        }
    }
    // Rights only mean anything with the rook still there — python-chess masks
    // against actual rooks, and a stale right would otherwise generate a move
    // that moves a piece which is not present.
    board.castling &= board.rooks;

    board.ep_square = parts.get(3).and_then(|s| {
        if *s == "-" {
            None
        } else {
            square_from_name(s)
        }
    });
    board.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    board.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    Ok(board)
}

pub fn to_fen(board: &Board) -> String {
    let mut out = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let sq = (rank * 8 + file) as u8;
            match board.piece_type_at(sq) {
                None => empty += 1,
                Some(piece) => {
                    if empty > 0 {
                        out.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let ch = match piece {
                        PAWN => 'p',
                        KNIGHT => 'n',
                        BISHOP => 'b',
                        ROOK => 'r',
                        QUEEN => 'q',
                        _ => 'k',
                    };
                    out.push(if board.color_at(sq) == Some(WHITE) {
                        ch.to_ascii_uppercase()
                    } else {
                        ch
                    });
                }
            }
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
        if rank > 0 {
            out.push('/');
        }
    }
    out.push(' ');
    out.push(if board.turn { 'w' } else { 'b' });
    out.push(' ');
    let mut rights = String::new();
    let clean = board.clean_castling();
    if clean & (1u64 << 7) != 0 {
        rights.push('K');
    }
    if clean & 1 != 0 {
        rights.push('Q');
    }
    if clean & (1u64 << 63) != 0 {
        rights.push('k');
    }
    if clean & (1u64 << 56) != 0 {
        rights.push('q');
    }
    out.push_str(if rights.is_empty() { "-" } else { &rights });
    out.push(' ');
    // `en_passant="legal"` is python-chess's default: an ep square with no
    // legal capture is not written.
    match board.ep_square {
        Some(sq) if board.has_legal_ep() => out.push_str(&square_name(sq)),
        _ => out.push('-'),
    }
    out.push_str(&format!(" {} {}", board.halfmove, board.fullmove));
    out
}

// ---------------------------------------------------------------------------
// Moves
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<u8>,
}

impl Move {
    pub fn uci(&self) -> String {
        let mut s = square_name(self.from);
        s.push_str(&square_name(self.to));
        if let Some(p) = self.promotion {
            s.push(match p {
                QUEEN => 'q',
                ROOK => 'r',
                BISHOP => 'b',
                KNIGHT => 'n',
                _ => '?',
            });
        }
        s
    }
}

const PROMOTION_PIECES: [u8; 4] = [QUEEN, ROOK, BISHOP, KNIGHT];

fn push_pawn_move(moves: &mut Vec<Move>, from: u8, to: u8) {
    let rank = rank_of(to);
    if rank == 0 || rank == 7 {
        for &p in PROMOTION_PIECES.iter() {
            moves.push(Move { from, to, promotion: Some(p) });
        }
    } else {
        moves.push(Move { from, to, promotion: None });
    }
}

/// Squares of a bitboard, highest index first — python-chess's `scan_reversed`.
///
/// Iteration order is not cosmetic here. `truncate_wins` returns the *first*
/// king capture found, so a port that scans low-to-high returns a different
/// winning move than the Python engine: same outcome, different recorded FEN
/// and different policy target. Caught in lockstep at ply 4,695, 2026-08-03.
#[inline]
fn scan_reversed(mut bb: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(bb.count_ones() as usize);
    while bb != 0 {
        let sq = (63 - bb.leading_zeros()) as u8;
        out.push(sq);
        bb &= !(1u64 << sq);
    }
    out
}

pub fn generate_pseudo_legal(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    let us = if board.turn { WHITE } else { BLACK };
    let them = if board.turn { BLACK } else { WHITE };
    let own = board.occupied_co[us];
    let enemy = board.occupied_co[them];
    let occ = board.occupied;

    // Section order below mirrors python-chess exactly: non-pawn pieces,
    // castling, pawn captures, pawn advances (all singles, then all doubles),
    // en passant.

    // --- non-pawn pieces -------------------------------------------------
    for from in scan_reversed(own & !board.pawns) {
        let piece = board.piece_type_at(from).unwrap();
        let attacks = match piece {
            KNIGHT => KNIGHT_ATTACKS[from as usize],
            KING => KING_ATTACKS[from as usize],
            BISHOP => sliding_attacks(from, occ, &BISHOP_DIRS),
            ROOK => sliding_attacks(from, occ, &ROOK_DIRS),
            QUEEN => {
                sliding_attacks(from, occ, &ROOK_DIRS) | sliding_attacks(from, occ, &BISHOP_DIRS)
            }
            _ => 0,
        };
        for to in scan_reversed(attacks & !own) {
            moves.push(Move { from, to, promotion: None });
        }
    }

    // --- castling --------------------------------------------------------
    // Check-aware, matching python-chess (see module docstring).
    let back_rank = if board.turn { 0u8 } else { 7u8 };
    if let Some(king_sq) = board.king_square(us) {
        if rank_of(king_sq) == back_rank && !board.is_attacked_by(them, king_sq) {
            for rook_sq in scan_reversed(board.clean_castling() & board.occupied_co[us]) {
                if rank_of(rook_sq) != back_rank {
                    continue;
                }
                let kingside = file_of(rook_sq) > file_of(king_sq);
                let king_to = if kingside { back_rank * 8 + 6 } else { back_rank * 8 + 2 };
                let rook_to = if kingside { back_rank * 8 + 5 } else { back_rank * 8 + 3 };

                // Every square the king and rook need must be empty, ignoring
                // the two pieces doing the castling.
                let ignore = (1u64 << king_sq) | (1u64 << rook_sq);
                let mut path_clear = true;
                for &(a, b) in &[(king_sq, king_to), (rook_sq, rook_to)] {
                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                    for sq in lo..=hi {
                        if occ & !ignore & (1u64 << sq) != 0 {
                            path_clear = false;
                            break;
                        }
                    }
                    if !path_clear {
                        break;
                    }
                }
                if !path_clear {
                    continue;
                }

                // The king may not pass through or land on an attacked square.
                let (lo, hi) = if king_sq <= king_to { (king_sq, king_to) } else { (king_to, king_sq) };
                let mut safe = true;
                for sq in lo..=hi {
                    if board.is_attacked_by(them, sq) {
                        safe = false;
                        break;
                    }
                }
                if safe {
                    moves.push(Move { from: king_sq, to: king_to, promotion: None });
                }
            }
        }
    }

    // --- pawn captures ---------------------------------------------------
    let pawns = own & board.pawns;
    for from in scan_reversed(pawns) {
        for to in scan_reversed(PAWN_ATTACKS[us][from as usize] & enemy) {
            push_pawn_move(&mut moves, from, to);
        }
    }

    // --- pawn advances: every single push, then every double -------------
    // python-chess builds both target bitboards up front and drains them in
    // that order, so a per-pawn loop would interleave them differently.
    let (singles, doubles) = if board.turn {
        let s1 = (pawns << 8) & !occ;
        let s2 = (s1 << 8) & !occ & ((0xFFu64 << 16) | (0xFFu64 << 24));
        (s1, s2)
    } else {
        let s1 = (pawns >> 8) & !occ;
        let s2 = (s1 >> 8) & !occ & ((0xFFu64 << 40) | (0xFFu64 << 32));
        (s1, s2)
    };
    for to in scan_reversed(singles) {
        let from = if board.turn { to - 8 } else { to + 8 };
        push_pawn_move(&mut moves, from, to);
    }
    for to in scan_reversed(doubles) {
        let from = if board.turn { to - 16 } else { to + 16 };
        moves.push(Move { from, to, promotion: None });
    }

    // --- en passant ------------------------------------------------------
    // Two constraints beyond "a pawn attacks the ep square", both from
    // python-chess and both load-bearing:
    //   * the capturer must stand on rank 5 (White) / rank 4 (Black) --
    //     `BB_RANKS[4 if turn else 3]`. Without it, White's own double push
    //     offers a capture of its own ep square (c2c4 then d2c3), because
    //     Monster Chess forces board.turn back to WHITE between halves.
    //     Caught by the API differential, 2026-08-03.
    //   * the ep square itself must be empty.
    if let Some(ep) = board.ep_square {
        if occ & (1u64 << ep) == 0 {
            let rank_mask: u64 = 0xFFu64 << (8 * if board.turn { 4 } else { 3 });
            let capturers = pawns & PAWN_ATTACKS[them][ep as usize] & rank_mask;
            for from in scan_reversed(capturers) {
                moves.push(Move { from, to: ep, promotion: None });
            }
        }
    }

    moves
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

#[pyfunction]
fn fen_roundtrip(fen: &str) -> PyResult<String> {
    parse_fen(fen)
        .map(|b| to_fen(&b))
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn pseudo_legal_uci(fen: &str) -> PyResult<Vec<String>> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    Ok(generate_pseudo_legal(&board).iter().map(|m| m.uci()).collect())
}

#[pyfunction]
fn attackers_of(fen: &str, by_white: bool, square: u8) -> PyResult<Vec<u8>> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    let color = if by_white { WHITE } else { BLACK };
    let mut bb = board.attackers(color, square);
    let mut out = Vec::new();
    while bb != 0 {
        out.push(bb.trailing_zeros() as u8);
        bb &= bb - 1;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_legal_uci, m)?)?;
    m.add_function(wrap_pyfunction!(attackers_of, m)?)?;
    Ok(())
}
