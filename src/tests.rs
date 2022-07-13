use std::collections::HashMap;

use ndarray::prelude::*;

use crate::load_checkpoint;

#[test]
fn test_rogue_net() {
    let rogue_net = load_checkpoint("norelpos/latest-step000000024576");
    let mut entities = HashMap::new();
    entities.insert("Head".to_string(), array![[3.0, 4.0]]);
    entities.insert("SnakeSegment".to_string(), array![[3.0, 4.0], [4.0, 4.0]]);
    entities.insert("Food".to_string(), array![[3.0, 5.0], [8.0, 4.0]]);
    let (probs, acts) = rogue_net.forward(&entities);
    let expected = array![[0.24504025, 0.24284518, 0.2426384, 0.26947618]];
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
