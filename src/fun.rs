use ndarray::prelude::*;

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
