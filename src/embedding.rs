use ndarray::prelude::*;

use crate::fun::{clip, relu};
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::msgpack::TensorDict;

#[derive(Debug, Clone)]
pub struct Embedding {
    mean: Array<f32, Ix2>,
    std: Array<f32, Ix2>,
    proj: Linear,
    ln: LayerNorm,
}

impl<'a> From<&'a TensorDict> for Embedding {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let norm = dict["0"].as_dict();
        let mean = norm["mean"]
            .as_tensor()
            .to_ndarray_f32()
            .insert_axis(Axis(0));
        let count = norm["count"].as_tensor().to_ndarray_f32();
        let squares_sum = norm["squares_sum"]
            .as_tensor()
            .to_ndarray_f32()
            .insert_axis(Axis(0));
        Embedding {
            std: (squares_sum / (count - 1.0))
                .mapv(|x| if x == 0.0 { 1.0 } else { x.sqrt() })
                .into_dimensionality()
                .unwrap(),
            mean: mean.into_dimensionality().unwrap(),
            proj: Linear::from(&dict["1"]),
            ln: LayerNorm::from(&dict["3"]),
        }
    }
}

impl Embedding {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let x = (&x - &self.mean) / &self.std;
        let x = clip(x.view(), -5.0, 5.0);
        let x = self.proj.forward(x.view());
        let x = relu(x.view());
        self.ln.forward(x.view())
    }
}
