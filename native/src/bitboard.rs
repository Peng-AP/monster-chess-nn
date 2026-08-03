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
    if board.castling & (1u64 << 7) != 0 {
        rights.push('K');
    }
    if board.castling & 1 != 0 {
        rights.push('Q');
    }
    if board.castling & (1u64 << 63) != 0 {
        rights.push('k');
    }
    if board.castling & (1u64 << 56) != 0 {
        rights.push('q');
    }
    out.push_str(if rights.is_empty() { "-" } else { &rights });
    out.push(' ');
    match board.ep_square {
        Some(sq) => out.push_str(&square_name(sq)),
        None => out.push('-'),
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

pub fn generate_pseudo_legal(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    let us = if board.turn { WHITE } else { BLACK };
    let them = if board.turn { BLACK } else { WHITE };
    let own = board.occupied_co[us];
    let enemy = board.occupied_co[them];
    let occ = board.occupied;

    // --- non-pawn pieces -------------------------------------------------
    let mut pieces = own & !board.pawns;
    while pieces != 0 {
        let from = pieces.trailing_zeros() as u8;
        pieces &= pieces - 1;
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
        let mut targets = attacks & !own;
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            moves.push(Move { from, to, promotion: None });
        }
    }

    // --- pawns -----------------------------------------------------------
    let mut pawns = own & board.pawns;
    let forward: i8 = if board.turn { 8 } else { -8 };
    let start_rank = if board.turn { 1 } else { 6 };
    while pawns != 0 {
        let from = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;

        let one = from as i8 + forward;
        if (0..64).contains(&one) && occ & (1u64 << one) == 0 {
            push_pawn_move(&mut moves, from, one as u8);
            let two = one + forward;
            if rank_of(from) == start_rank && (0..64).contains(&two) && occ & (1u64 << two) == 0 {
                moves.push(Move { from, to: two as u8, promotion: None });
            }
        }

        let mut caps = PAWN_ATTACKS[us][from as usize] & enemy;
        while caps != 0 {
            let to = caps.trailing_zeros() as u8;
            caps &= caps - 1;
            push_pawn_move(&mut moves, from, to);
        }

        // En passant. The square is whatever the FEN carries, which is exactly
        // the "conferred only by the last push" divergence: python-chess holds
        // one ep square and each push recomputes it.
        if let Some(ep) = board.ep_square {
            if PAWN_ATTACKS[us][from as usize] & (1u64 << ep) != 0 {
                moves.push(Move { from, to: ep, promotion: None });
            }
        }
    }

    // --- castling --------------------------------------------------------
    // Check-aware, matching python-chess (see module docstring).
    let back_rank = if board.turn { 0u8 } else { 7u8 };
    if let Some(king_sq) = board.king_square(us) {
        if rank_of(king_sq) == back_rank && !board.is_attacked_by(them, king_sq) {
            let mut rights = board.castling & board.occupied_co[us];
            while rights != 0 {
                let rook_sq = rights.trailing_zeros() as u8;
                rights &= rights - 1;
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
