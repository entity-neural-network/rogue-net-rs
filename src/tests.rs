use std::collections::HashMap;

use ndarray::prelude::*;

use crate::rogue_net::{FwdArgs, RogueNet};

#[test]
fn test_vanilla_rogue_net() {
    test_snake_net(
        "test-data/simple",
        array![[0.2477651, 0.24801087, 0.25257978, 0.25164425]],
    );
    test_snake_net(
        "test-data/relpos-encoding",
        array![[0.24349655, 0.2540197, 0.25025287, 0.25223088]],
    );
}

fn test_snake_net(checkpoint: &str, expected: Array2<f32>) {
    let rogue_net = RogueNet::load(checkpoint);
    let mut entities = HashMap::new();
    entities.insert("Head".to_string(), array![[3.0, 4.0]]);
    entities.insert("SnakeSegment".to_string(), array![[3.0, 4.0], [4.0, 4.0]]);
    entities.insert("Food".to_string(), array![[3.0, 5.0], [8.0, 4.0]]);
    let (probs, acts) = rogue_net.forward(FwdArgs {
        features: entities,
        actors: vec!["Head".to_string()],
    });
    assert_eq!(acts.len(), 1);
    assert!(acts[0] < 4);
    assert!(
        probs.abs_diff_eq(&expected, 1e-6),
        "{:?} != {:?}\n{:?}",
        probs,
        expected,
        &probs - &expected
    );
}
