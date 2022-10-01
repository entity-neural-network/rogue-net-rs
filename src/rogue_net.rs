use indexmap::IndexMap;
use ndarray::{concatenate, s, Array2, Axis};
use ron::extensions::Extensions;
use std::collections::HashMap;
use std::env;
use std::fmt::Write;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::categorical_action_head::CategoricalActionHead;
use crate::config::RogueNetConfig;
use crate::config::TrainConfig;
use crate::embedding::Embedding;
use crate::msgpack::decode_state_dict;
use crate::msgpack::TensorDict;
use crate::state::{ObsSpace, State};
use crate::transformer::Transformer;

#[derive(Debug, Clone)]
/// Implements the [RogueNet](https://github.com/entity-neural-network/rogue-net) entity neural network.
pub struct RogueNet {
    pub config: RogueNetConfig,
    pub obs_space: ObsSpace,
    translation: Option<Translate>,
    embeddings: Vec<(String, Embedding)>,
    backbone: Transformer,
    action_heads: IndexMap<String, CategoricalActionHead>,
}

#[derive(Debug, Clone, Default)]
/// Arguments for RogueNet forward pass.
pub struct FwdArgs {
    pub features: HashMap<String, Array2<f32>>,
    pub actors: Vec<String>,
}

#[derive(Debug, Clone)]
struct Translate {
    reference_entity: String,
    rotation_vec_indices: Option<[usize; 2]>,
    position_feature_indices: HashMap<String, Vec<usize>>,
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
    /// use rogue_net::{RogueNet, FwdArgs};
    ///
    /// let rogue_net = RogueNet::load("test-data/simple");
    /// let mut features = HashMap::new();
    /// features.insert("Head".to_string(), array![[3.0, 4.0]]);
    /// features.insert("SnakeSegment".to_string(), array![[3.0, 4.0], [4.0, 4.0]]);
    /// features.insert("Food".to_string(), array![[3.0, 5.0], [8.0, 4.0]]);
    /// let (action_probs, actions) = rogue_net.forward(FwdArgs { features, ..Default::default() });
    /// ```
    pub fn forward(&self, mut args: FwdArgs) -> (Array2<f32>, Vec<u64>) {
        if env::var("ROGUE_NET_DUMP_INPUTS").is_ok() {
            args.dump(self.action_heads.get_index(0).unwrap().0)
                .unwrap();
        }

        if let Some(t) = &self.translation {
            let reference_entity = args
                .features
                .get(&t.reference_entity)
                .unwrap_or_else(|| panic!("Missing entity type: {}", t.reference_entity));
            let origin = t.position_feature_indices[&t.reference_entity]
                .iter()
                .map(|&i| reference_entity[[0, i]])
                .collect::<Vec<_>>();
            let rotation = t
                .rotation_vec_indices
                .map(|r| (reference_entity[[0, r[0]]], reference_entity[[0, r[1]]]));
            for (entity, feats) in args.features.iter_mut() {
                if *entity != t.reference_entity {
                    for i in 0..feats.dim().0 {
                        match rotation {
                            Some((rx, ry)) => {
                                let x =
                                    feats[[i, t.position_feature_indices[entity][0]]] - origin[0];
                                let y =
                                    feats[[i, t.position_feature_indices[entity][1]]] - origin[1];
                                feats[[i, t.position_feature_indices[entity][0]]] = x * rx + y * ry;
                                feats[[i, t.position_feature_indices[entity][1]]] =
                                    -x * ry + y * rx;
                            }
                            None => {
                                for (j, x) in
                                    t.position_feature_indices[entity].iter().zip(origin.iter())
                                {
                                    feats[[i, *j]] -= x;
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut actors = vec![];
        let mut i = 0;
        let mut embeddings = Vec::with_capacity(args.features.len());
        for (key, embedding) in &self.embeddings {
            let x = embedding.forward(args.features[key].view());
            if args.actors.iter().any(|a| a == key) {
                for j in i..i + x.dim().0 {
                    actors.push(j);
                }
            }
            i += x.dim().0;
            embeddings.push(x);
        }
        let x = concatenate(
            Axis(0),
            &embeddings.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let x = self.backbone.forward(x, &args.features);
        self.action_heads
            .values()
            .next()
            .unwrap()
            .forward(x.view(), actors)
    }

    fn new(state_dict: &TensorDict, config: RogueNetConfig, state: &State) -> Self {
        assert!(
            config.embd_pdrop == 0.0 && config.resid_pdrop == 0.0 && config.attn_pdrop == 0.0,
            "dropout is not supported"
        );
        assert!(config.pooling.is_none(), "pooling is not supported");

        let translation = config.translation.as_ref().map(|t| {
            assert!(
                t.rotation_angle_feature.is_none(),
                "rotation_angle_feature not implemented",
            );
            assert!(!t.add_dist_feature, "add_dist_features not implemented");
            let rotation_vec_indices = t.rotation_vec_features.as_ref().map(|rot| {
                let indices = rot
                    .iter()
                    .map(|s| {
                        state.obs_space.entities[&t.reference_entity]
                            .features
                            .iter()
                            .position(|f| f == s)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(indices.len(), 2, "rotation_vec_features must have length 2");
                [indices[0], indices[1]]
            });
            let position_feature_indices = state
                .obs_space
                .entities
                .iter()
                .map(|(name, entity)| {
                    let indices = t
                        .position_features
                        .iter()
                        .map(|f| {
                            entity
                                .features
                                .iter()
                                .position(|f2| f2 == f)
                                .unwrap_or_else(|| {
                                    panic!("feature \"{}\" not found in reference entity", f)
                                })
                        })
                        .collect::<Vec<_>>();
                    (name.clone(), indices)
                })
                .collect();
            Translate {
                reference_entity: t.reference_entity.clone(),
                rotation_vec_indices,
                position_feature_indices,
            }
        });

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
            translation,
            backbone,
            action_heads,
            config,
            obs_space: state.obs_space.clone(),
        }
    }

    /// Adapts the RogueNet neural network to the given observation space by
    /// filtering out any features that were not present during training.
    pub fn with_obs_filter(mut self, obs_space: HashMap<String, Vec<String>>) -> Self {
        for (entity, received_features) in obs_space {
            if let Some((_, embedding)) = self.embeddings.iter_mut().find(|(e, _)| *e == entity) {
                embedding.set_obs_filter(
                    &self.obs_space.entities[&entity].features,
                    &received_features,
                );
            }
        }
        self
    }
}

impl FwdArgs {
    fn dump(&self, action_name: &str) -> Result<(), std::fmt::Error> {
        let mut out = String::new();
        writeln!(out, "obs = Observation(")?;

        // Features
        writeln!(out, "    features={{")?;
        for (entity_name, features) in &self.features {
            writeln!(out, "        \"{entity_name}\": [")?;
            for i in 0..features.dim().0 {
                writeln!(out, "            {},", features.slice(s![i, ..]))?;
            }
            writeln!(out, "        ],")?;
        }
        writeln!(out, "    }},")?;

        // IDs
        writeln!(out, "    ids={{")?;
        let mut total = 0;
        for (entity_name, features) in &self.features {
            let count = features.dim().0;
            writeln!(
                out,
                "        \"{entity_name}\": {:?},",
                &(total..total + count).collect::<Vec<_>>()[..]
            )?;
            total += count;
        }
        writeln!(out, "    }},")?;

        // done, reward
        writeln!(out, "    done=False,")?;
        writeln!(out, "    reward=0.0,")?;

        // Actions
        writeln!(
            out,
            "    actions={{\"{action_name}\": CategoricalActionMask(actor_types={:?})}},",
            &self.actors[..]
        )?;

        writeln!(out, ")")?;

        println!("{}", out);

        Ok(())
    }
}
