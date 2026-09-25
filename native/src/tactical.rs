//! Exact immediate-turn king-capture queries, without enumerating every second
//! move. After each possible first move, an attack on the king is equivalent
//! to a legal pseudo-capture: king capture ends the game before own safety.
use crate::bitboard::{Board,WHITE,BLACK,generate_pseudo_legal};
use crate::game::Game;

fn white_capture(board: &Board, pending: bool) -> bool {
    let king=match board.king_square(BLACK) {Some(k)=>k,None=>return false};
    if board.king_square(WHITE).is_none() {return false;}
    if board.is_attacked_by(WHITE,king) {return true;}
    if pending {return false;}
    let mut before=board.clone(); before.turn=true;
    for mv in generate_pseudo_legal(&before) {
        let mut after=before.clone(); after.push(&mv);
        if after.king_square(BLACK).is_none() || after.is_attacked_by(WHITE,king) {return true;}
    }
    false
}

fn black_capture(board: &Board) -> bool {
    if board.king_square(BLACK).is_none() {return false;}
    board.king_square(WHITE).is_some_and(|k| board.is_attacked_by(BLACK,k))
}

pub fn capture_in_turn(g: &Game) -> bool {
    if g.is_white_turn {white_capture(g.board_ref(),g.white_half_pending)}
    else {black_capture(g.board_ref())}
}

pub fn mover_under_threat(g: &Game) -> bool {
    if g.is_white_turn {black_capture(g.board_ref())}
    else {white_capture(g.board_ref(),false)}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_queries_match_existing_scans_on_ten_thousand_walk_positions() {
        let start="rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1";
        let mut g=Game::from_fen(start).unwrap();
        let mut rng=193u64;
        for _ in 0..10_000 {
            if g.is_terminal_rust() {g=Game::from_fen(start).unwrap();}
            let old=crate::eval::pre_nn_clamp(g.board_ref(),g.is_white_turn,g.white_half_pending).is_some();
            assert_eq!(capture_in_turn(&g),old,"{} pending{}",g.fen_string(),g.white_half_pending);
            let old_threat=if g.is_white_turn {crate::eval::black_threat_scan(g.board_ref())}
                else {crate::eval::white_threat_scan(g.board_ref(),false)};
            assert_eq!(mover_under_threat(&g),old_threat,"{}",g.fen_string());
            let actions=g.search_actions_rust();
            if actions.is_empty() {g=Game::from_fen(start).unwrap();continue;}
            rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            g.apply_half(&actions[(rng>>32) as usize%actions.len()]).unwrap();
        }
    }
    #[test]
    fn promotions_and_discovered_attacks_are_included() {
        for fen in ["7k/3P4/8/8/8/8/8/K7 w - - 0 1",
                    "7k/8/8/8/8/8/6B1/K5R1 w - - 0 1",
                    "8/8/8/8/4k3/8/4K3/8 w - - 0 1"] {
            let g=Game::from_fen(fen).unwrap();
            assert_eq!(capture_in_turn(&g),crate::eval::white_threat_scan(g.board_ref(),false));
        }
    }
}
