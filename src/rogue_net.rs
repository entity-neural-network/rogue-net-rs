use std::collections::HashMap;

use indexmap::IndexMap;
use ndarray::{concatenate, Array2, Axis};

use crate::categorical_action_head::CategoricalActionHead;
use crate::config::RogueNetConfig;
use crate::embedding::Embedding;
use crate::msgpack::TensorDict;
use crate::state::State;
use crate::transformer::Transformer;

#[derive(Debug, Clone)]
pub struct RogueNet {
    pub config: RogueNetConfig,
    embeddings: Vec<(String, Embedding)>,
    backbone: Transformer,
    action_heads: IndexMap<String, CategoricalActionHead>,
}

impl RogueNet {
    pub fn forward(&self, entities: &HashMap<String, Array2<f32>>) -> (Array2<f32>, Vec<u64>) {
        let mut embeddings = Vec::with_capacity(entities.len());
        for (key, embedding) in &self.embeddings {
            let x = embedding.forward(entities[key].view());
            embeddings.push(x);
        }
        let x = concatenate(
            Axis(0),
            &embeddings.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let x = self.backbone.forward(x, entities);
        self.action_heads
            .values()
            .next()
            .unwrap()
            .forward(x.view(), vec![0])
    }

    pub fn new(state_dict: &TensorDict, config: RogueNetConfig, state: &State) -> Self {
        assert!(
            config.embd_pdrop == 0.0 && config.resid_pdrop == 0.0 && config.attn_pdrop == 0.0,
            "dropout is not supported"
        );
        assert!(config.pooling.is_none(), "pooling is not supported");
        assert!(config.translation.is_none(), "translation is not supported");

        let dict = state_dict.as_dict();
        let mut embeddings = Vec::new();
        for (key, value) in dict["embedding"].as_dict()["embeddings"].as_dict() {
            let embedding = Embedding::from(value);
            embeddings.push((key.clone(), embedding));
        }
        let backbone = Transformer::new(&dict["backbone"], &config, state);

        let mut action_heads = IndexMap::new();
        for (key, value) in dict["action_heads"].as_dict() {
            let action_head = CategoricalActionHead::from(value);
            action_heads.insert(key.clone(), action_head);
        }

        RogueNet {
            embeddings,
            backbone,
            action_heads,
            config,
        }
    }
}
