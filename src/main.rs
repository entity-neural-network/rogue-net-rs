mod embedding;
mod fun;
mod layer_norm;
mod linear;
mod msgpack;
mod rogue_net;

use std::collections::HashMap;

use ndarray::prelude::*;

use crate::msgpack::decode_state_dict;
use crate::rogue_net::RogueNet;

fn main() {
    //let bytes = include_bytes!("test.msgpack");
    let bytes = include_bytes!("../checkpoints/latest-step000000024576/state.agent.msgpack");
    let state_dict = decode_state_dict(bytes).unwrap();
    // println!("{:#?}", state_dict);
    let rogue_net = RogueNet::from(&state_dict);
    println!("{:#?}", rogue_net);
    let mut entities = HashMap::new();
    entities.insert("Head".to_string(), array![[3.0, 4.0]]);
    entities.insert("SnakeSegment".to_string(), array![[3.0, 4.0], [4.0, 4.0]]);
    entities.insert("Food".to_string(), array![[3.0, 5.0], [8.0, 4.0]]);
    rogue_net.forward(&entities);
}
