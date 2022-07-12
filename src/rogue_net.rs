use std::collections::HashMap;

use ndarray::{concatenate, Array2, Axis};

use crate::embedding::Embedding;
use crate::msgpack::TensorDict;

#[derive(Debug, Clone)]
pub struct RogueNet {
    embeddings: HashMap<String, Embedding>,
}

impl RogueNet {
    pub fn forward(&self, entities: &HashMap<String, Array2<f32>>) -> Array2<f32> {
        let mut embeddings = Vec::with_capacity(entities.len());
        for (key, entity) in entities {
            let x = self.embeddings[key].forward(entity.view());
            println!("{} {:?}", key, x);
            embeddings.push(x);
        }
        concatenate(
            Axis(0),
            &embeddings.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap()
    }
}

impl<'a> From<&'a TensorDict> for RogueNet {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let mut embeddings = HashMap::new();
        for (key, value) in dict["embedding"].as_dict()["embeddings"].as_dict() {
            let embedding = Embedding::from(value);
            embeddings.insert(key.clone(), embedding);
        }

        RogueNet { embeddings }
    }
}
