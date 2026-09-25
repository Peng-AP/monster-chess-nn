//! Optional diagnostic reservoir of actual, uncached neural evaluations.
//! Opt-in instrumentation, disabled by default. No search
//! ordering randomness: this RNG belongs only to sampling, with its own seed.
use crate::game::Game;
use crate::bitboard::square_name_pub;
use crate::bitboard::Move;

pub type Sample=(String,bool,u32,f64);
pub struct Recorder {
    limit: usize,
    rng: u64,
    pub seen: u64,
    samples: Vec<(Game,f64,Vec<Move>)>,
}

impl Recorder {
    pub fn new(limit: usize,seed: u64)->Self {
        Self {limit,rng:seed,seen:0,samples:Vec::with_capacity(limit)}
    }
    fn below(&mut self,n: u64)->u64 {
        // Rejection eliminates modulo bias. Not a cryptographic generator.
        let threshold=n.wrapping_neg()%n;
        loop {
            self.rng=self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if self.rng>=threshold {return self.rng%n;}
        }
    }
    #[inline]
    pub fn observe(&mut self,g: &Game,raw_value: f64) {
        self.observe_with_path(g,raw_value,&[]);
    }
    pub fn observe_with_path(&mut self,g: &Game,raw_value: f64,path: &[Move]) {
        if self.limit==0 {return;}
        self.seen+=1;
        let index=if self.samples.len()<self.limit {self.samples.len() as u64}
                  else {self.below(self.seen)};
        if index<self.limit as u64 {
            let sample=(g.clone(),raw_value,path.to_vec());
            if index==self.samples.len() as u64 {self.samples.push(sample);}
            else {self.samples[index as usize]=sample;}
        }
    }
    pub fn paths(&self)->Vec<Vec<String>> {
        self.samples.iter().map(|(_,_,path)|path.iter().map(|m|m.uci()).collect()).collect()
    }
    pub fn finish(self)->Vec<Sample> {
        self.samples.into_iter().map(|(g,v,_)| {
            // Normal FEN/repetition identity can suppress raw EP. Evaluation
            // inputs need the actual board field, even when no legal EP exists.
            let fen=g.fen_string();
            let mut fields:Vec<_>=fen.split_whitespace().map(str::to_owned).collect();
            fields[3]=g.board_ref().ep_square.map_or_else(||"-".into(),square_name_pub);
            (fields.join(" "),g.white_half_pending,g.turn_count,v)
        }).collect()
    }
}

pub fn raw_fen(g: &Game)->String {
    let fen=g.fen_string();
    let mut fields:Vec<_>=fen.split_whitespace().map(str::to_owned).collect();
    fields[3]=g.board_ref().ep_square.map_or_else(||"-".into(),square_name_pub);
    fields.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_sampling_is_empty_and_enabled_is_bounded_reproducible() {
        let g=Game::from_fen("k7/8/8/8/8/8/3P4/4K3 w - - 0 1").unwrap();
        let mut a=Recorder::new(7,9175);let mut b=Recorder::new(7,9175);
        let mut off=Recorder::new(0,9175);
        for i in 0..1000 {a.observe(&g,i as f64);b.observe(&g,i as f64);off.observe(&g,i as f64);}
        assert_eq!(a.seen,1000);assert_eq!(off.seen,0);
        let samples=a.finish();assert_eq!(samples.len(),7);assert_eq!(samples,b.finish());
        assert!(off.finish().is_empty());
    }
    #[test]
    fn sample_preserves_raw_ep_not_just_repetition_fen() {
        let g=Game::from_state("k7/8/8/8/3P4/8/8/4K3 b - d3 0 1",false,1).unwrap();
        let mut recorder=Recorder::new(1,0);recorder.observe(&g,0.25);
        let sample=recorder.finish().pop().unwrap();
        assert_eq!(sample.0.split_whitespace().nth(3),Some("d3"));
        assert_eq!(sample.2,1);assert_eq!(sample.3,0.25);
    }
}
