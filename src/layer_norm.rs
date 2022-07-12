use ndarray::prelude::*;

use crate::msgpack::TensorDict;

#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Array<f32, Ix1>,
    bias: Array<f32, Ix1>,
}

impl<'a> From<&'a TensorDict> for LayerNorm {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let weight = dict["weight"].as_tensor().to_ndarray_f32();
        let bias = dict["bias"].as_tensor().to_ndarray_f32();
        LayerNorm {
            weight: weight.into_dimensionality().unwrap(),
            bias: bias.into_dimensionality().unwrap(),
        }
    }
}

impl LayerNorm {
    pub fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let std = (&x - &mean).std_axis(Axis(1), 0.0).insert_axis(Axis(1));
        (x - mean) / (std + 1e-5) * &self.weight + &self.bias
    }
}
