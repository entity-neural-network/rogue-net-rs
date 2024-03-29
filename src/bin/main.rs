use clap::Parser;
use rogue_net::RogueNet;

use std::fs::File;
use std::path::PathBuf;
use tar::Builder;

#[derive(Parser)]
#[clap(name = "rogue-net-cli")]
#[clap(bin_name = "rogue-net-cli")]
enum Cmd {
    Archive(Pack),
    Check(Check),
}

#[derive(clap::Args)]
#[clap(author, version, about, long_about = None)]
struct Pack {
    /// Path to checkpoint file to archive
    #[clap(value_parser)]
    path: PathBuf,
}

#[derive(clap::Args)]
#[clap(author, version, about, long_about = None)]
struct Check {
    /// Path to archive to check
    #[clap(value_parser)]
    path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cmd::parse() {
        Cmd::Archive(Pack { path }) => {
            // Append .tar to the path
            let file = File::create(path.with_extension("roguenet"))?;
            let mut a = Builder::new(file);

            a.append_path_with_name(path.join("config.ron"), "config.ron")
                .unwrap();
            a.append_path_with_name(path.join("state.ron"), "state.ron")
                .unwrap();
            a.append_path_with_name(path.join("state.agent.msgpack"), "state.agent.msgpack")
                .unwrap();
            a.finish()?;
        }
        Cmd::Check(Check { path }) => {
            RogueNet::load_archive(
                File::open(path.clone())
                    .map_err(|e| format!("Failed to open archive {}: {}", path.display(), e))?,
            )?;
        }
    }

    Ok(())
}
