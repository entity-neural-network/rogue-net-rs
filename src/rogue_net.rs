use indexmap::IndexMap;
use ndarray::{concatenate, Array2, Axis};
use ron::extensions::Extensions;
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
/// Implements the [RogueNet](https://github.com/entity-neural-network/rogue-net) entity neural network.
pub struct RogueNet {
    pub config: RogueNetConfig,
    embeddings: Vec<(String, Embedding)>,
    backbone: Transformer,
    action_heads: IndexMap<String, CategoricalActionHead>,
}

impl RogueNet {
    /// Loads the parameters for a trained RogueNet neural network from a checkpoint directory produced by [enn-trainer](https://github.com/entity-neural-network/enn-trainer).
    ///
    /// # Arguments
    /// * `path` - Path to the checkpoint directory.
    pub fn load<P: AsRef<Path>>(path: P) -> RogueNet {
        let config_path = path.as_ref().join("config.ron");
        let ron = ron::Options::default().with_default_extension(Extensions::IMPLICIT_SOME);

        let config: TrainConfig = ron
            .from_reader(
                File::open(&config_path)
                    .unwrap_or_else(|_| panic!("Failed to open {}", config_path.display())),
            )
            .unwrap();

        let state_path = path.as_ref().join("state.ron");
        let state: State = ron
            .from_reader(
                File::open(&state_path)
                    .unwrap_or_else(|_| panic!("Failed to open {}", state_path.display())),
            )
            .unwrap();

        let agent_path = path.as_ref().join("state.agent.msgpack");
        let state_dict = decode_state_dict(File::open(&agent_path).unwrap()).unwrap();
        RogueNet::new(&state_dict, config.net, &state)
    }

    /// Loads the parameters for a trained RogueNet neural network from a tar archive of a checkpoint directory.
    ///
    /// # Arguments
    /// * `r` - A reader for the tar archive.
    ///
    /// # Example
    /// ```
    /// use std::fs::File;
    /// use rogue_net::RogueNet;
    ///
    /// let rogue_net = RogueNet::load_archive(File::open("test-data/simple.roguenet").unwrap());
    /// ```
    pub fn load_archive<R: Read>(r: R) -> Result<RogueNet, std::io::Error> {
        let mut a = tar::Archive::new(r);
        let mut config: Option<TrainConfig> = None;
        let mut state = None;
        let mut state_dict = None;
        let ron = ron::Options::default().with_default_extension(Extensions::IMPLICIT_SOME);
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
                "config.ron" => config = Some(ron.from_reader(file).unwrap()),
                "state.ron" => state = Some(ron.from_reader(file).unwrap()),
                "state.agent.msgpack" => state_dict = Some(decode_state_dict(file).unwrap()),
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unexpected file: {}", file.path().unwrap().display()),
                    ))
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

    /// Runs a forward pass of the RogueNet neural network.
    ///
    /// # Arguments
    /// * `entities` - Maps each entity type to an `Array2<f32>` containing the entities' features.
    ///
    /// # Example
    /// ```
    /// use std::collections::HashMap;
    /// use ndarray::prelude::*;
    /// use rogue_net::RogueNet;
    ///
    /// let rogue_net = RogueNet::load("test-data/simple");
    /// let mut entities = HashMap::new();
    /// entities.insert("Head".to_string(), array![[3.0, 4.0]]);
    /// entities.insert("SnakeSegment".to_string(), array![[3.0, 4.0], [4.0, 4.0]]);
    /// entities.insert("Food".to_string(), array![[3.0, 5.0], [8.0, 4.0]]);
    /// let (action_probs, actions) = rogue_net.forward(&entities);
    /// ```
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

    fn new(state_dict: &TensorDict, config: RogueNetConfig, state: &State) -> Self {
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
