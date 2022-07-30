use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct State {
    pub step: u32,
    pub restart: u32,
    pub next_eval_step: Option<u32>,
    pub agent: String,
    pub value_function: Option<String>,
    pub optimizer: String,
    pub vf_optimizer: Option<String>,
    pub obs_space: ObsSpace,
    pub action_space: IndexMap<String, ActionSpace>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ObsSpace {
    pub global_features: Vec<String>,
    pub entities: IndexMap<String, Entity>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Entity {
    pub features: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum ActionSpace {
    CategoricalActionSpace { index_to_label: Vec<String> },
}
