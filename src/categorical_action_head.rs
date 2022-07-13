use ndarray::prelude::*;
use rand::Rng;

use crate::fun::softmax;
use crate::linear::Linear;
use crate::msgpack::TensorDict;
#[derive(Debug, Clone)]
pub struct CategoricalActionHead {
    proj: Linear,
}

impl<'a> From<&'a TensorDict> for CategoricalActionHead {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        CategoricalActionHead {
            proj: Linear::from(&dict["proj"]),
        }
    }
}

impl CategoricalActionHead {
    pub fn forward(&self, x: ArrayView2<f32>, actors: Vec<usize>) -> (Array2<f32>, Vec<u64>) {
        let actor_x = x.select(Axis(0), &actors);
        let logits = self.proj.forward(actor_x.view());
        let probs = softmax(&logits);
        // TODO: efficient sampling
        let mut rng = rand::thread_rng();
        let mut acts = vec![0; actors.len()];
        for i in 0..probs.dim().0 {
            let mut r = rng.gen::<f32>();
            for j in 0..probs.dim().1 {
                r -= probs[[i, j]];
                if r <= 0.0 {
                    acts[i] = j as u64;
                    break;
                }
            }
        }
        (probs, acts)
    }
}
