//! Portable independent accumulators let LLVM vectorize a dense dot product.
//! Reassociation changes FP32 rounding, so validate against training and record
//! maximum error before using in search. Not quantization or a new model.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(),b.len());
    let mut sums=[0.0f32;8];
    let blocks=a.len()/8;
    for block in 0..blocks {
        let start=block*8;
        for lane in 0..8 {sums[lane]+=a[start+lane]*b[start+lane];}
    }
    let mut sum=sums.iter().copied().sum::<f32>();
    for i in blocks*8..a.len() {sum+=a[i]*b[i];}
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_accumulators_remain_close_to_high_precision_reference() {
        let mut state=17u64;
        for n in [1,3,8,32,128,512,1024] {
            let mut random=|| {state=state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state>>32) as u32 as f64/u32::MAX as f64-0.5) as f32};
            let a:Vec<_>=(0..n).map(|_|random()).collect();
            let b:Vec<_>=(0..n).map(|_|random()).collect();
            let reference:f64=a.iter().zip(&b).map(|(&a,&b)|a as f64*b as f64).sum();
            assert!((dot(&a,&b) as f64-reference).abs()<1e-5);
        }
    }
}
