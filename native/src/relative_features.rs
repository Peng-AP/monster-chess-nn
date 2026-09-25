//! MCSV002 offsets: two king centers x twelve piece types/colors x 15x15.
//! Absolute840 stays present; no color reversal, move advice or tactical labels.
use crate::bitboard::{WHITE,BLACK};
use crate::game::Game;

pub const INPUTS: usize=6240;

pub fn visit(g: &Game, mut add: impl FnMut(usize,f32)) {
    let b=g.board_ref();
    for (center,color) in [WHITE,BLACK].iter().enumerate() {
        let king=match b.king_square(*color) {Some(s)=>s,None=>continue};
        let mut occupied=b.occupied;
        while occupied!=0 {
            let sq=occupied.trailing_zeros() as u8;
            occupied&=occupied-1;
            let offset=if b.occupied_co[WHITE]&(1u64<<sq)!=0 {0} else {6};
            let piece=offset+b.piece_type_at(sq).unwrap() as usize-1;
            let dx=(sq%8) as i32-(king%8) as i32+7;
            let dy=(sq/8) as i32-(king/8) as i32+7;
            add(840+center*2700+piece*225+dy as usize*15+dx as usize,1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_offsets_survive_translation() {
        let a=Game::from_fen("8/8/8/8/4k3/8/3P4/2K5 w - - 0 1").unwrap();
        let b=Game::from_fen("8/8/8/8/5k2/8/4P3/3K4 w - - 0 1").unwrap();
        let mut fa=vec![];let mut fb=vec![];
        visit(&a,|i,_| fa.push(i));visit(&b,|i,_| fb.push(i));
        fa.sort();fb.sort();assert_eq!(fa,fb);
        assert!(fa.contains(&(840+8*15+8)));
        assert!(fa.contains(&(840+2700+5*15+6)));
        assert!(fa.iter().all(|&i|i<INPUTS));
    }
}
