//! Conservative minimax transposition bounds. Exact remaining depth and exact
//! settled-path suffix are part of the key. The pre-root repetition history is
//! constant for this per-search cache and is not duplicated in every entry.
//! More restrictive than a normal chess TT, intentionally: no graph-history
//! aliasing, no deeper-entry reuse that changes fixed-depth minimax semantics.
use std::collections::HashMap;
use crate::game::Game;
use crate::bitboard::Move;
use crate::search_order::PositionKey;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct BoundKey(PositionKey, u32, u32, u32, Vec<PositionKey>);
impl BoundKey {
    pub fn new(g: &Game, depth: u32, extension_budget: u32, path_suffix: &[PositionKey]) -> Self {
        Self(PositionKey::new(g),g.turn_count,depth,extension_budget,path_suffix.to_vec())
    }
}

#[derive(Clone, Copy)]
enum Bound { Exact, Upper, Lower }
struct Entry { value: f64, action: Option<Move>, bound: Bound }
pub struct ScoreCache {
    table: HashMap<BoundKey,Entry>, capacity: usize,
    pub hits: u64, pub cutoffs: u64, pub rejected: u64,
}
impl ScoreCache {
    pub fn new() -> Self { Self::with_capacity(32_768) }
    pub fn with_capacity(capacity: usize) -> Self {
        Self { table: HashMap::new(), capacity, hits: 0, cutoffs: 0, rejected: 0 }
    }
    /// Retain allocated buckets and lifetime telemetry; discard depth-specific bounds.
    pub fn next_iteration(&mut self) { self.table.clear(); }
    pub fn probe(&mut self, key: &BoundKey, alpha: &mut f64, beta: &mut f64)
        -> Option<(f64,Option<Move>)>
    {
        let e=self.table.get(key)?;
        self.hits+=1;
        match e.bound {
            Bound::Exact => { self.cutoffs+=1; return Some((e.value,e.action.clone())); },
            Bound::Upper => *beta=beta.min(e.value),
            Bound::Lower => *alpha=alpha.max(e.value),
        }
        if *alpha>=*beta { self.cutoffs+=1; Some((e.value,e.action.clone())) } else { None }
    }
    pub fn store(&mut self, key: BoundKey, value: f64, action: Option<Move>, alpha: f64, beta: f64) {
        if self.table.len()>=self.capacity && !self.table.contains_key(&key) {
            self.rejected+=1; return;
        }
        let bound=if value<=alpha { Bound::Upper } else if value>=beta { Bound::Lower } else { Bound::Exact };
        self.table.insert(key,Entry{value,action,bound});
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn history_depth_and_extensions_cannot_alias() {
        let g=Game::from_fen("k7/8/8/8/8/8/3P4/4K3 w - - 0 1").unwrap();
        let key=BoundKey::new(&g,2,0,&[]);
        let mut c=ScoreCache::new();
        c.store(key.clone(),0.2,Some(Move{from:11,to:19,promotion:None}),-2.0,2.0);
        assert!(c.probe(&key,&mut -2.0,&mut 2.0).is_some());
        for other in [BoundKey::new(&g,3,0,&[]),BoundKey::new(&g,2,1,&[]),
                      BoundKey::new(&g,2,0,&[PositionKey::new(&g)])] {
            assert!(c.probe(&other,&mut -2.0,&mut 2.0).is_none());
        }
    }
    #[test]
    fn fail_high_bound_is_not_an_exact_value() {
        let g=Game::from_fen("k7/8/8/8/8/8/3P4/4K3 w - - 0 1").unwrap();
        let key=BoundKey::new(&g,2,0,&[]);
        let mut c=ScoreCache::new();
        c.store(key.clone(),0.5,Some(Move{from:11,to:19,promotion:None}),-1.0,0.4);
        let (mut alpha,mut beta)=(-1.0,1.0);
        assert!(c.probe(&key,&mut alpha,&mut beta).is_none());
        assert_eq!(alpha,0.5);
        assert_eq!(beta,1.0);
        assert!(c.probe(&key,&mut -1.0,&mut 0.4).is_some());
    }
}
