//! Search-first ordering only. Nothing here changes the legal move set or
//! allows a score cutoff. Board transpositions deliberately carry NO values:
//! identical boards reached through different repetition histories can differ.
use std::collections::HashMap;
use crate::game::Game;
use crate::bitboard::Move;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct PositionKey([u64; 10], bool, bool);

impl PositionKey {
    pub fn new(g: &Game) -> Self {
        let b = g.board_ref();
        Self([b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
              b.occupied_co[0], b.occupied_co[1], b.castling,
              b.ep_square.map_or(64, |s| s as u64)], g.is_white_turn, g.white_half_pending)
    }
    pub fn repetition(g: &Game) -> Self {
        let mut key=Self::new(g);
        // Match the driver's python-chess FEN identity: only legal EP is
        // represented for repetition. Search/evaluation keys retain raw EP.
        if !g.board_ref().has_legal_ep() { key.0[9]=64; }
        key
    }
}

pub fn move_id(m: &Move) -> usize {
    // Promotions are distinct TT/killer moves. History intentionally shares
    // from/to credit; it is only a heuristic ordering hint.
    let promotion=m.promotion.map_or(0, |p| p as usize);
    promotion*4096 + m.from as usize*64 + m.to as usize
}

pub struct Ordering {
    table: HashMap<PositionKey, usize>,
    history: Vec<i32>,
    killers: Vec<[Option<usize>; 2]>,
}

impl Ordering {
    pub fn new() -> Self {
        Self { table: HashMap::new(), history: vec![0; 3*4096], killers: vec![[None;2]; 512] }
    }
    fn phase(g: &Game) -> usize {
        if !g.is_white_turn { 2 } else if g.white_half_pending { 1 } else { 0 }
    }
    pub fn sort(&self, g: &Game, moves: &mut [Move], ply: usize) {
        let preferred = self.table.get(&PositionKey::new(g)).copied();
        let killers = self.killers.get(ply).copied().unwrap_or([None;2]);
        let base = Self::phase(g)*4096;
        moves.sort_unstable_by_key(|m| {
            let id = move_id(m);
            let to = (id%4096%64) as u8;
            let capture = g.board_ref().piece_type_at(to).unwrap_or(0) as i64;
            let score = if preferred == Some(id) { 1_000_000_000 }
                else if killers[0] == Some(id) { 10_000_000 }
                else if killers[1] == Some(id) { 9_000_000 }
                else { capture*100_000 + self.history[base + id%4096] as i64
                     + if id>=4096 { 50_000 } else { 0 } };
            (-score,id)
        });
    }
    pub fn best(&mut self, g: &Game, mv: &Move) {
        let key = PositionKey::new(g);
        if self.table.len()<200_000 || self.table.contains_key(&key) {
            self.table.insert(key, move_id(mv));
        }
    }
    /// Root-only learned ordering, with the previous iteration's best first.
    /// Stable sorting retains existing capture/history order for unlisted moves.
    pub fn root_hint(&self,g: &Game,moves: &mut [Move],hint: &[Move]) {
        let preferred=self.table.get(&PositionKey::new(g)).copied();
        moves.sort_by_key(|m| {
            if preferred==Some(move_id(m)) {0}
            else {1+hint.iter().position(|h| h==m).unwrap_or(hint.len())}
        });
    }
    pub fn cutoff(&mut self, g: &Game, mv: &Move, ply: usize, depth: u32) {
        let id = move_id(mv);
        if g.board_ref().piece_type_at((id%64) as u8).is_some() { return; }
        if let Some(k) = self.killers.get_mut(ply) {
            if k[0] != Some(id) { *k = [Some(id), k[0]]; }
        }
        let value = &mut self.history[Self::phase(g)*4096 + id%4096];
        let bonus = (depth*depth).min(400) as i32;
        *value += bonus - *value*bonus/100_000;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_and_raw_ep_are_distinct_but_clocks_are_not() {
        let a = Game::from_state("k7/8/8/8/8/8/3P4/4K3 w - - 0 1",false,0).unwrap();
        let b = Game::from_state("k7/8/8/8/8/8/3P4/4K3 w - - 4 7",false,10).unwrap();
        let c = Game::from_state("k7/8/8/8/8/8/3P4/4K3 w - - 0 1",true,0).unwrap();
        assert!(PositionKey::new(&a)==PositionKey::new(&b));
        assert!(PositionKey::new(&a)!=PositionKey::new(&c));
        let e = Game::from_state("k7/8/8/8/8/8/3P4/4K3 w - e6 0 1",false,0).unwrap();
        assert!(PositionKey::new(&a)!=PositionKey::new(&e));
    }
    #[test]
    fn ordering_preserves_all_moves_and_promotion_identity() {
        let g = Game::from_fen("k7/8/8/8/8/8/3P4/4K3 w - - 0 1").unwrap();
        let mut o = Ordering::new();
        let mut actions = crate::monster::white_single_moves(g.board_ref());
        let preferred = *actions.last().unwrap();
        let mut original = actions.clone(); original.sort_by_key(move_id);
        o.best(&g,&preferred);
        o.sort(&g,&mut actions,0);
        assert_eq!(actions[0],preferred);
        actions.sort_by_key(move_id); assert_eq!(actions,original);
        assert_ne!(move_id(&Move{from:48,to:56,promotion:Some(5)}),
                   move_id(&Move{from:48,to:56,promotion:Some(2)}));
    }
}
