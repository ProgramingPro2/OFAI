use ofai_sim::capabilities::Capabilities;
use ofai_sim::protocol::serve;
use ofai_sim::terrain::TerrainCache;

fn main() {
    let mut port = 8765u16;
    let mut maps = TerrainCache::default_maps_dir();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" => {
                port = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(port);
            }
            "--maps-dir" => {
                if let Some(p) = args.next() {
                    maps = p.into();
                }
            }
            "--capabilities" => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Capabilities::foundation()).unwrap()
                );
                return;
            }
            "--help" => {
                eprintln!(
                    "ofai-sim [--port 8765] [--maps-dir PATH] [--capabilities]\n\
                     Schema-v2 TCP env server. Unsupported later features fail closed."
                );
                return;
            }
            _ => {}
        }
    }
    if let Err(e) = serve(port, maps) {
        eprintln!("[ofai-sim] fatal: {e}");
        std::process::exit(1);
    }
}
