//! Training-only complete-turn minimax plans for batched GPU frontier values.
//! No change to the playing searches. Partial/capped trees never yield labels.
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;
use crate::game::{Game,MAX_GAME_TURNS};
use crate::bitboard::{Move,WHITE,BLACK};
use crate::search_order::PositionKey;

struct Node { game:Game, children:Vec<usize>, value:Option<f64>, actions:Vec<String> }
#[pyclass]
pub struct LabelTree {
    nodes:Vec<Node>,frontier:Vec<usize>,records:Vec<usize>,limit:usize,root_count:u32,
    #[pyo3(get)] repetition_hits:usize,
    #[pyo3(get)] cap_hits:usize,
}
fn actions(g:&Game)->Vec<Move> {
    if !g.is_white_turn {crate::monster::black_actions(g.board_ref(),true)}
    else if g.white_half_pending {crate::monster::white_second_half_moves(g.board_ref())}
    else {crate::monster::white_single_moves(g.board_ref())}
}
impl LabelTree {
    fn build(&mut self,g:Game,depth:u32,counts:&mut HashMap<PositionKey,usize>,path:Vec<String>)
        ->Result<usize,String> {
        if self.nodes.len()>=self.limit {return Err("label tree node limit exceeded; no partial targets".into());}
        let idx=self.nodes.len();let settled=!g.white_half_pending;
        let key=PositionKey::repetition(&g);
        let terminal=if g.board_ref().king_square(WHITE).is_none() {Some(-1.)}
            else if g.board_ref().king_square(BLACK).is_none() {Some(1.)}
            else if g.turn_count>=MAX_GAME_TURNS {self.cap_hits+=1;Some(0.)}
            else if settled && counts.get(&key).copied().unwrap_or(0)>=2 {
                self.repetition_hits+=1;Some(0.)
            } else {None};
        if idx==0 || (settled && g.turn_count==self.root_count+1) {
            self.records.push(idx);
        }
        self.nodes.push(Node {game:g.clone(),children:vec![],value:terminal,actions:path.clone()});
        if terminal.is_some() {return Ok(idx);}
        if depth==0 && settled {
            if crate::tactical::capture_in_turn(&g) {
                self.nodes[idx].value=Some(if g.is_white_turn {1.} else {-1.});
            } else {self.frontier.push(idx);}
            return Ok(idx);
        }
        if settled {*counts.entry(key).or_insert(0)+=1;}
        for mv in actions(&g) {
            let mut child=g.clone();child.apply_half_move(&mv,false);
            let mut next=path.clone();next.push(mv.uci());
            let depth=if g.turn_completing() {depth.saturating_sub(1)} else {depth};
            let c=self.build(child,depth,counts,next)?;
            self.nodes[idx].children.push(c);
        }
        if settled {
            let n=counts.get_mut(&key).unwrap();*n-=1;
            if *n==0 {counts.remove(&key);}
        }
        if self.nodes[idx].children.is_empty() {self.nodes[idx].value=Some(0.);}
        Ok(idx)
    }
}
#[pymethods]
impl LabelTree {
    #[new]
    #[pyo3(signature=(fen,pending=false,turn_count=0,prior_positions=None,depth=2,node_limit=250000))]
    fn new(py:Python<'_>,fen:&str,pending:bool,turn_count:u32,prior_positions:Option<Vec<String>>,
           depth:u32,node_limit:usize)->PyResult<Self> {
        if !(1..=3).contains(&depth) || !(1..=1000000).contains(&node_limit) {
            return Err(PyValueError::new_err("depth1..3, node_limit1..1000000 required"));
        }
        let game=Game::from_state(fen,pending,turn_count).map_err(PyValueError::new_err)?;
        let mut counts=HashMap::new();
        for fen in prior_positions.unwrap_or_default() {
            if fen.split_whitespace().count()!=4 {return Err(PyValueError::new_err("Expected four-field prior keys excluding root"));}
            let g=Game::from_fen(&format!("{fen} 0 1")).map_err(PyValueError::new_err)?;
            *counts.entry(PositionKey::repetition(&g)).or_insert(0)+=1;
        }
        py.detach(move || {
            let mut tree=Self {nodes:vec![],frontier:vec![],records:vec![],limit:node_limit,
                               root_count:turn_count,repetition_hits:0,cap_hits:0};
            tree.build(game,depth,&mut counts,vec![]).map_err(PyValueError::new_err)?;
            Ok(tree)
        })
    }
    fn node_count(&self)->usize {self.nodes.len()}
    fn frontier_count(&self)->usize {self.frontier.len()}
    fn record_count(&self)->usize {self.records.len()}
    fn record_states(&self)->Vec<(usize,String,bool,u32,Option<f64>,Vec<String>)> {
        self.records.iter().map(|&i| {
            let n=&self.nodes[i];(i,crate::leaf_recorder::raw_fen(&n.game),n.game.white_half_pending,
                                  n.game.turn_count,n.value,n.actions.clone())
        }).collect()
    }
    /// NHWC float32 input bytes and fixed-White perspective multipliers.
    #[pyo3(signature=(start=0,batch=512,channels=17,records=false))]
    fn input_batch<'py>(&self,py:Python<'py>,start:usize,batch:usize,channels:usize,records:bool)
        ->PyResult<(Bound<'py,PyBytes>,Vec<f64>)> {
        if batch==0 || batch>4096 {return Err(PyValueError::new_err("batch1..4096 required"));}
        let ids=if records {&self.records} else {&self.frontier};
        if start>ids.len() {return Err(PyValueError::new_err("start outside input list"));}
        let mut bytes=Vec::new();let mut signs=Vec::new();
        for &i in &ids[start..ids.len().min(start+batch)] {
            let g=&self.nodes[i].game;
            let encoded=crate::encoding::encode_with_turn(g.board_ref(),g.is_white_turn,
                g.white_half_pending,channels,Some(g.turn_count)).map_err(PyValueError::new_err)?;
            bytes.extend(encoded.iter().flat_map(|x|x.to_le_bytes()));
            signs.push(if g.is_white_turn {1.} else {-1.});
        }
        Ok((PyBytes::new(py,&bytes),signs))
    }
    /// Return fixed-White values for record_states(), without modifying the tree.
    fn solve(&self,values:Vec<f64>)->PyResult<Vec<f64>> {
        if values.len()!=self.frontier.len() || values.iter().any(|v|!v.is_finite() || v.abs()>1.00001) {
            return Err(PyValueError::new_err("Expected one finite White value in [-1,1] per frontier"));
        }
        let mut scores:Vec<_>=self.nodes.iter().map(|n|n.value).collect();
        for (&i,v) in self.frontier.iter().zip(values) {scores[i]=Some(v.clamp(-0.99,0.99));}
        for i in (0..self.nodes.len()).rev() {
            if scores[i].is_none() {
                let n=&self.nodes[i];
                let mut value=if n.game.is_white_turn {-2.0f64} else {2.0f64};
                for &c in &n.children {
                    let v=scores[c].expect("Children are computed before parents");
                    value=if n.game.is_white_turn {value.max(v)} else {value.min(v)};
                }
                scores[i]=Some(value);
            }
        }
        Ok(self.records.iter().map(|&i|scores[i].unwrap()).collect())
    }
}
pub fn register(m:&Bound<'_,PyModule>)->PyResult<()> {
    m.add_class::<LabelTree>()?;Ok(())
}
