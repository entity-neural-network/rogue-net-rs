mod categorical_action_head;
mod embedding;
mod fun;
mod layer_norm;
mod linear;
mod msgpack;
mod rogue_net;
#[cfg(test)]
mod tests;
mod transformer;

use std::path::Path;

pub use crate::rogue_net::RogueNet;
use msgpack::decode_state_dict;

pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> RogueNet {
    let bytes = std::fs::read(path).unwrap();
    let state_dict = decode_state_dict(&bytes).unwrap();
    RogueNet::from(&state_dict)
}
