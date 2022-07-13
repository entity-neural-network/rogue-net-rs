mod categorical_action_head;
mod embedding;
mod fun;
mod layer_norm;
mod linear;
mod msgpack;
mod rogue_net;
mod transformer;

use std::path::Path;

use msgpack::decode_state_dict;
pub use rogue_net::RogueNet;

pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> RogueNet {
    let bytes = std::fs::read(path).unwrap();
    let state_dict = decode_state_dict(&bytes).unwrap();
    RogueNet::from(&state_dict)
}
