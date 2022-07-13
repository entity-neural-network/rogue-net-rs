use ndarray::prelude::*;
use statrs::function::erf::erf;

pub fn relu(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|x| if x < 0.0 { 0.0 } else { x })
}

pub fn clip(x: ArrayView2<f32>, min: f32, max: f32) -> Array2<f32> {
    x.mapv(|x| {
        if x < min {
            min
        } else if x > max {
            max
        } else {
            x
        }
    })
}

pub fn gelu(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|x| 0.5 * x * (1.0 + erf((x / std::f32::consts::SQRT_2) as f64)) as f32)
}

pub fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut softmax = logits.to_owned();
    // Calculate softmax
    let max = softmax.fold_axis(Axis(1), 0.0, |x, y| if *x > *y { *x } else { *y });
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x = (*x - max[b]).exp();
    }
    let sum = softmax.sum_axis(Axis(1));
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x /= sum[b];
    }
    softmax
}
