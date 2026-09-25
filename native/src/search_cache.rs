//! Static evaluation memoization, NOT a cache of searched minimax bounds.
//! The caller MUST check king capture, turn cap, and path-dependent repetition
//! before looking here. Network inputs depend on board/phase/remaining budget,
//! not on history. Exact current-turn capture scans share this same identity.
use std::collections::HashMap;
use crate::game::Game;
use crate::search_order::PositionKey;

pub struct EvalCache {
    table: HashMap<(PositionKey,u32), f64>,
    pub hits: u64,
    pub misses: u64,
}

impl EvalCache {
    pub fn new() -> Self { Self { table: HashMap::new(), hits: 0, misses: 0 } }
    pub fn get(&mut self, g: &Game) -> Option<f64> {
        let answer=self.table.get(&(PositionKey::new(g),g.turn_count)).copied();
        if answer.is_some() { self.hits+=1; } else { self.misses+=1; }
        answer
    }
    pub fn put(&mut self, g: &Game, value: f64) {
        if self.table.len()<200_000 {
            self.table.insert((PositionKey::new(g),g.turn_count),value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remaining_budget_and_phase_are_not_interchangeable() {
        let fen="k7/8/8/8/8/8/3P4/4K3 w - - 0 1";
        let a=Game::from_state(fen,false,0).unwrap();
        let b=Game::from_state(fen,false,1).unwrap();
        let c=Game::from_state(fen,true,0).unwrap();
        let mut cache=EvalCache::new();
        cache.put(&a,0.25);
        assert_eq!(cache.get(&a),Some(0.25));
        assert_eq!(cache.get(&b),None);
        assert_eq!(cache.get(&c),None);
    }
}
