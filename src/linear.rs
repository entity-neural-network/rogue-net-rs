use ndarray::prelude::*;

use crate::msgpack::TensorDict;
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Array<f32, Ix2>,
    bias: Array<f32, Ix2>,
}

impl<'a> From<&'a TensorDict> for Linear {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let weight = dict["weight"].as_tensor().to_ndarray_f32();
        let bias = dict["bias"].as_tensor().to_ndarray_f32();
        Linear {
            weight: weight.reversed_axes().into_dimensionality().unwrap(),
            bias: bias.insert_axis(Axis(0)).into_dimensionality().unwrap(),
        }
    }
}

impl Linear {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        x.dot(&self.weight) + &self.bias
    }
}
