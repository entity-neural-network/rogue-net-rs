use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{concatenate, s, Array2, ArrayView2, Axis};

use crate::config::RogueNetConfig;
use crate::fun::{gelu, softmax};
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::msgpack::TensorDict;
use crate::relpos_encoding::{RelposEncoding, RelposIndices};
use crate::state::State;

#[derive(Debug, Clone)]
pub struct Transformer {
    relpos_encoding: Option<Arc<RelposEncoding>>,
    blocks: Vec<TransformerBlock>,
}

impl Transformer {
    pub fn forward(
        &self,
        mut x: Array2<f32>,
        entities: &HashMap<String, Array2<f32>>,
    ) -> Array2<f32> {
        let relpos_indices = self
            .relpos_encoding
            .as_ref()
            .map(|rp| rp.relpos_indices(entities));
        log::debug!("relpos_indices: {:?}", relpos_indices);

        for block in &self.blocks {
            x = block.forward(x, &relpos_indices);
        }
        x
    }

    pub fn new(state_dict: &TensorDict, config: &RogueNetConfig, state: &State) -> Self {
        let dict = state_dict.as_dict();

        let relpos_encoding = config.relpos_encoding.clone().map(|config| {
            Arc::new(RelposEncoding::new(
                &dict["relpos_encoding"],
                &config,
                &state.obs_space,
            ))
        });

        let mut blocks = Vec::new();
        for value in dict["blocks"].as_dict().values() {
            let block = TransformerBlock::new(value, config.n_head, &relpos_encoding);
            blocks.push(block);
        }

        Transformer {
            blocks,
            relpos_encoding,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    ln1: LayerNorm,
    attention: MultiHeadAttention,
    ln2: LayerNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    pub fn forward(&self, x: Array2<f32>, relpos_indices: &Option<RelposIndices>) -> Array2<f32> {
        let x0 = x.view();
        let x = self.ln1.forward(x.view());
        let x = self.attention.forward(x.view(), relpos_indices);
        let x = x + x0;
        log::debug!("ATTN + RESIDUAL {:?}", x);
        let x1 = x.view();
        let x = self.ln2.forward(x.view());
        let x = self.mlp.forward(x);
        log::debug!("MLP {:?}", x);
        let x = x + x1;
        log::debug!("MLP + RESIDUAL {:?}", x);
        x
    }

    fn new(
        state_dict: &TensorDict,
        n_head: u32,
        relpos_encoding: &Option<Arc<RelposEncoding>>,
    ) -> Self {
        let dict = state_dict.as_dict();
        let ln1 = LayerNorm::from(&dict["ln1"]);
        let mlp = Mlp::from(&dict["mlp"]);
        let ln2 = LayerNorm::from(&dict["ln2"]);
        let attention = MultiHeadAttention::new(&dict["attn"], n_head, relpos_encoding.clone());

        TransformerBlock {
            ln1,
            mlp,
            ln2,
            attention,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    n_head: u32,
    relpos_encoding: Option<Arc<RelposEncoding>>,
    key: Linear,
    value: Linear,
    query: Linear,
    proj: Linear,
}

impl MultiHeadAttention {
    pub fn forward(
        &self,
        x: ArrayView2<f32>,
        relpos_indices: &Option<RelposIndices>,
    ) -> Array2<f32> {
        let (_, c) = x.dim();
        let d_head = c / self.n_head as usize;
        let k = self.key.forward(x);
        let q = self.query.forward(x);
        let v = self.value.forward(x);
        let scale = 1.0 / (d_head as f32).sqrt();
        let mut ys = vec![];
        for head in 0..self.n_head as usize {
            let slice = s![.., head * d_head..(head + 1) * d_head];
            let q = q.slice(slice);
            let k = k.slice(slice);
            let mut logits = q.dot(&k.t());
            logits.mapv_inplace(|x| x * scale);
            if let Some(re) = &self.relpos_encoding {
                let relattn_logits = &re.relattn_logits(relpos_indices.as_ref().unwrap(), q.view());
                logits += relattn_logits;
            }
            let attn = softmax(&logits);
            let v = v.slice(slice);
            let mut y = attn.dot(&v);
            if let Some(re) = &self.relpos_encoding {
                let relpos_values = &re.relpos_values(relpos_indices.as_ref().unwrap(), &attn, x);
                log::debug!("RELPOS VALUES {:?}", relpos_values);
                y += relpos_values;
            }
            ys.push(y);
        }
        let y = concatenate(Axis(1), &ys.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
        self.proj.forward(y.view())
    }
    fn new(
        state_dict: &TensorDict,
        n_head: u32,
        relpos_encoding: Option<Arc<RelposEncoding>>,
    ) -> Self {
        let dict = state_dict.as_dict();
        let key = Linear::from(&dict["key"]);
        let value = Linear::from(&dict["value"]);
        let query = Linear::from(&dict["query"]);
        let proj = Linear::from(&dict["proj"]);

        MultiHeadAttention {
            relpos_encoding,
            n_head,
            key,
            value,
            query,
            proj,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mlp {
    layer1: Linear,
    layer2: Linear,
}

impl Mlp {
    pub fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        let x = self.layer1.forward(x.view());
        let x = gelu(x.view());
        self.layer2.forward(x.view())
    }
}

impl<'a> From<&'a TensorDict> for Mlp {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let layer1 = Linear::from(&dict["0"]);
        let layer2 = Linear::from(&dict["2"]);

        Mlp { layer1, layer2 }
    }
}
