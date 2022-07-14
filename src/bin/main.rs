use std::collections::HashMap;

use ndarray::prelude::*;

use rogue_net::load_checkpoint;

fn main() {
    env_logger::init();
    let rogue_net = load_checkpoint("checkpoints/latest-step000000024576");
    let mut entities = HashMap::new();
    entities.insert("Head".to_string(), array![[3.0, 4.0]]);
    entities.insert("SnakeSegment".to_string(), array![[3.0, 4.0], [4.0, 4.0]]);
    entities.insert("Food".to_string(), array![[3.0, 5.0], [8.0, 4.0]]);
    let (probs, acts) = rogue_net.forward(&entities);
    dbg!(probs, acts);
}
