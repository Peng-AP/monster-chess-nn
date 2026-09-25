//! Opt-in, disjoint local cost counters. Never time recursive child visits.
use std::time::Instant;
pub const LABELS: [&str; 8] = ["static_cache", "tactical", "evaluation", "move_generation",
    "ordering", "clone_apply", "bound_lookup", "bound_store"];
pub struct Profile { pub enabled: bool, pub nanos: [u64; 8], pub calls: [u64; 8] }
impl Profile {
    pub fn new(enabled: bool) -> Self { Self { enabled, nanos: [0; 8], calls: [0; 8] } }
    #[inline] pub fn start(&self) -> Option<Instant> { self.enabled.then(Instant::now) }
    #[inline] pub fn end(&mut self, category: usize, start: Option<Instant>) {
        if let Some(t) = start { self.nanos[category] += t.elapsed().as_nanos() as u64;
            self.calls[category] += 1; }
    }
}
