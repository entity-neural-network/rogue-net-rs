use indexmap::IndexMap;
use ndarray::{concatenate, Array2, Axis};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::categorical_action_head::CategoricalActionHead;
use crate::config::RogueNetConfig;
use crate::config::TrainConfig;
use crate::embedding::Embedding;
use crate::msgpack::decode_state_dict;
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
    pub fn load<P: AsRef<Path>>(path: P) -> RogueNet {
        let config_path = path.as_ref().join("config.ron");
        let config: TrainConfig = ron::de::from_reader(
            File::open(&config_path)
                .unwrap_or_else(|_| panic!("Failed to open {}", config_path.display())),
        )
        .unwrap();

        let state_path = path.as_ref().join("state.ron");
        let state: State = ron::de::from_reader(
            File::open(&state_path)
                .unwrap_or_else(|_| panic!("Failed to open {}", state_path.display())),
        )
        .unwrap();

        let agent_path = path.as_ref().join("state.agent.msgpack");
        let state_dict = decode_state_dict(File::open(&agent_path).unwrap()).unwrap();
        RogueNet::new(&state_dict, config.net, &state)
    }

    pub fn load_archive<R: Read>(r: R) -> Result<RogueNet, Box<dyn std::error::Error>> {
        let mut a = tar::Archive::new(r);
        let mut config: Option<TrainConfig> = None;
        let mut state = None;
        let mut state_dict = None;
        for file in a.entries()? {
            let file = file?;
            match file
                .path()?
                .components()
                .last()
                .unwrap()
                .as_os_str()
                .to_str()
                .unwrap()
            {
                "config.ron" => config = Some(ron::de::from_reader(file).unwrap()),
                "state.ron" => state = Some(ron::de::from_reader(file).unwrap()),
                "state.agent.msgpack" => state_dict = Some(decode_state_dict(file).unwrap()),
                _ => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Unexpected file: {}", file.path().unwrap().display()),
                    )))
                }
            }
        }
        Ok(RogueNet::new(
            &state_dict.ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::Other, "Missing state.agent.msgpack")
            })?,
            config
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::Other, "Missing config.ron")
                })?
                .net,
            &state.ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::Other, "Missing state.ron")
            })?,
        ))
    }

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
