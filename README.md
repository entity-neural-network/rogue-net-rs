# RogueNet Rust

[![Crates.io](https://img.shields.io/crates/v/rogue-net.svg)](https://crates.io/crates/rogue-net)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](./LICENSE)
[![Crates.io](https://img.shields.io/crates/d/rogue-net.svg)](https://crates.io/crates/rogue-net)
[![Actions Status](https://github.com/entity-neural-network/rogue-net-rs/workflows/Test/badge.svg)](https://github.com/entity-neural-network/rogue-net-rs/actions)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)

The rogue-net crate provides a pure Rust implementation of the [RogueNet neural network](https://github.com/entity-neural-network/rogue-net).
It can be used to load agents created with the [Entity Neural Network Trainer](https://github.com/entity-neural-network/enn-trainer) and use them inside Rust applications.

```rust
use std::collections::HashMap;
use ndarray::prelude::*;
use rogue_net::RogueNet;

let rogue_net = RogueNet::load("checkpoint-dir");
let mut entities = HashMap::new();
entities.insert("Head".to_string(), array![[3.0, 4.0]]);
entities.insert("SnakeSegment".to_string(), array![[3.0, 4.0], [4.0, 4.0]]);
entities.insert("Food".to_string(), array![[3.0, 5.0], [8.0, 4.0]]);
let (action_probs, actions) = rogue_net.forward(&entities);
```
