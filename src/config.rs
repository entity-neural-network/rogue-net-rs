use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct TrainConfig {
    pub version: u32,
    pub env: EnvConfig,
    pub net: RogueNetConfig,
    pub optim: OptimizerConfig,
    pub ppo: PPOConfig,
    pub rollout: RolloutConfig,
    pub eval: Option<EvalConfig>,
    pub vf_net: Option<RogueNetConfig>,
    pub name: String,
    pub seed: u64,
    pub total_timesteps: u64,
    pub max_train_time: Option<u64>,
    pub torch_deterministic: bool,
    pub cuda: bool,
    pub track: bool,
    pub wandb_project_name: String,
    pub wandb_entity: String,
    pub capture_samples: Option<u64>,
    pub capture_logits: bool,
    pub capture_samples_subsample: u64,
    pub trial: Option<String>,
    pub data_dir: String,
    pub cuda_empty_cache: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvConfig {
    pub kwargs: String,
    pub id: String,
    pub validate: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RogueNetConfig {
    pub embd_pdrop: f64,
    pub resid_pdrop: f64,
    pub attn_pdrop: f64,
    pub n_layer: u32,
    pub n_head: u32,
    pub d_model: u32,
    pub pooling: Option<String>,
    pub relpos_encoding: MaybeRelpos,
    pub d_qk: u32,
    pub translation: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OptimizerConfig {
    pub lr: f64,
    pub bs: u32,
    pub weight_decay: f64,
    pub micro_bs: Option<u32>,
    pub anneal_lr: bool,
    pub update_epochs: u32,
    pub max_grad_norm: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PPOConfig {
    pub gae: bool,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub norm_adv: bool,
    pub clip_coef: f64,
    pub clip_vloss: bool,
    pub ent_coef: f64,
    pub vf_coef: f64,
    pub target_kl: Option<f64>,
    pub anneal_entropy: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RolloutConfig {
    pub steps: u32,
    pub num_envs: u32,
    pub processes: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EvalConfig;

// Hack to allow RelposEncodingConfig to be either "None" or "RelposEncodingConfig" (rather than "Some(RelposEncodingConfig)")
#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum MaybeRelpos {
    None,
    RelposEncodingConfig {
        extent: Vec<u32>,
        position_features: Vec<String>,
        scale: f32,
        per_entity_values: bool,
        exclude_entities: Vec<String>,
        value_relpos_projection: bool,
        key_relpos_projection: bool,
        per_entity_projections: bool,
        radial: bool,
        distance: bool,
        rotation_vec_features: Option<Vec<String>>,
        rotation_angle_feature: Option<String>,
        interpolate: bool,
        value_gate: String,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RelposEncodingConfig {
    pub extent: Vec<u32>,
    pub position_features: Vec<String>,
    pub scale: f32,
    pub per_entity_values: bool,
    pub exclude_entities: Vec<String>,
    pub value_relpos_projection: bool,
    pub key_relpos_projection: bool,
    pub per_entity_projections: bool,
    pub radial: bool,
    pub distance: bool,
    pub rotation_vec_features: Option<Vec<String>>,
    pub rotation_angle_feature: Option<String>,
    pub interpolate: bool,
    pub value_gate: String,
}
