use ndarray::{concatenate, s, Array2, ArrayView2, Axis};

use crate::fun::{gelu, softmax};
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::msgpack::TensorDict;

#[derive(Debug, Clone)]
pub struct Transformer {
    blocks: Vec<TransformerBlock>,
}

impl Transformer {
    pub fn forward(&self, mut x: Array2<f32>) -> Array2<f32> {
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

impl<'a> From<&'a TensorDict> for Transformer {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let mut blocks = Vec::new();
        for value in dict["blocks"].as_dict().values() {
            let block = TransformerBlock::from(value);
            blocks.push(block);
        }

        Transformer { blocks }
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
    pub fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        let x0 = x.view();
        let x = self.ln1.forward(x.view());
        let x = self.attention.forward(x.view());
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
}

impl<'a> From<&'a TensorDict> for TransformerBlock {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let ln1 = LayerNorm::from(&dict["ln1"]);
        let mlp = Mlp::from(&dict["mlp"]);
        let ln2 = LayerNorm::from(&dict["ln2"]);
        let attention = MultiHeadAttention::from(&dict["attn"]);

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
    key: Linear,
    value: Linear,
    query: Linear,
    proj: Linear,
}

impl MultiHeadAttention {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let (_, c) = x.dim();
        // TODO: read from config
        let n_head = 2;
        let d_head = c / n_head;
        let k = self.key.forward(x);
        let q = self.query.forward(x);
        let v = self.value.forward(x);
        let scale = 1.0 / (d_head as f32).sqrt();
        let mut ys = vec![];
        for head in 0..n_head {
            let slice = s![.., head * d_head..(head + 1) * d_head];
            let q = q.slice(slice);
            let k = k.slice(slice);
            let mut logits = q.dot(&k.t());
            logits.mapv_inplace(|x| x * scale);
            let attn = softmax(&logits);
            let v = v.slice(slice);
            let y = attn.dot(&v);
            ys.push(y);
        }
        let y = concatenate(Axis(1), &ys.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
        self.proj.forward(y.view())
    }
}

impl<'a> From<&'a TensorDict> for MultiHeadAttention {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let key = Linear::from(&dict["key"]);
        let value = Linear::from(&dict["value"]);
        let query = Linear::from(&dict["query"]);
        let proj = Linear::from(&dict["proj"]);

        MultiHeadAttention {
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
