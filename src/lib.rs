mod categorical_action_head;
mod config;
mod embedding;
mod fun;
mod layer_norm;
mod linear;
mod msgpack;
mod relpos_encoding;
mod rogue_net;
mod state;
#[cfg(test)]
mod tests;
mod transformer;

use std::path::Path;

pub use crate::rogue_net::RogueNet;
use config::TrainConfig;
use msgpack::decode_state_dict;
use state::State;

pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> RogueNet {
    let config_path = path.as_ref().join("config.ron");
    let config: TrainConfig = ron::de::from_reader(
        std::fs::File::open(&config_path)
            .unwrap_or_else(|_| panic!("Failed to open {}", config_path.display())),
    )
    .unwrap();

    let state_path = path.as_ref().join("state.ron");
    let state: State = ron::de::from_reader(
        std::fs::File::open(&state_path)
            .unwrap_or_else(|_| panic!("Failed to open {}", state_path.display())),
    )
    .unwrap();

    let agent_path = path.as_ref().join("state.agent.msgpack");
    let bytes = std::fs::read(agent_path).unwrap();
    let state_dict = decode_state_dict(&bytes).unwrap();
    RogueNet::new(&state_dict, config.net, &state)
}
