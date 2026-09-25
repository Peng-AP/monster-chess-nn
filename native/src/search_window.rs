//! Principal-variation search windows. This is exact alpha-beta, not depth
//! reduction or speculative move pruning. Every potential in-window improvement
//! must be re-searched at the original window. White max/max needs no negation.
pub fn scout(maximizing: bool,alpha: f64,beta: f64)->(f64,f64) {
    debug_assert!(alpha<beta);
    if maximizing {(alpha,alpha.next_up().min(beta))}
    else {(beta.next_down().max(alpha),beta)}
}

pub fn needs_research(value: f64,alpha: f64,beta: f64)->bool {
    value>alpha && value<beta
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scout_windows_are_strict_and_stay_inside_parent_window() {
        for (a,b) in [(-2.0,2.0),(-1.0,-0.99),(-0.1,0.0),(0.0,0.1),(0.99,1.0)] {
            for side in [false,true] {
                let (x,y)=scout(side,a,b);
                assert!(a<=x && x<y && y<=b);
            }
        }
        assert!(!needs_research(-0.5,-0.5,0.5));
        assert!(!needs_research(0.5,-0.5,0.5));
        assert!(needs_research(0.0,-0.5,0.5));
    }
}
