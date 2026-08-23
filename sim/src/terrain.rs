//! Training-safe terrain loader: cache immutable bins, build a fresh map per game.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::map::GameMap;
use crate::spec::MapSize;

#[derive(Clone, Debug, Deserialize)]
pub struct MapMetadata {
    pub width: u32,
    pub height: u32,
    pub num_land_tiles: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ManifestNation {
    pub name: String,
    pub coordinates: Option<[u32; 2]>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MapManifest {
    pub name: String,
    pub map: MapMetadata,
    pub map4x: MapMetadata,
    pub map16x: MapMetadata,
    #[serde(default)]
    pub nations: Vec<ManifestNation>,
}

#[derive(Clone)]
struct CachedBins {
    manifest: MapManifest,
    map_bin: Vec<u8>,
    map4x_bin: Vec<u8>,
    map16x_bin: Vec<u8>,
}

pub struct FreshTerrain {
    pub game_map: GameMap,
    pub mini_map: GameMap,
    pub manifest: MapManifest,
    pub nations: Vec<ManifestNation>,
}

pub struct TerrainCache {
    maps_dir: PathBuf,
    bins: HashMap<String, CachedBins>,
}

impl TerrainCache {
    pub fn new(maps_dir: impl Into<PathBuf>) -> Self {
        Self {
            maps_dir: maps_dir.into(),
            bins: HashMap::new(),
        }
    }

    pub fn default_maps_dir() -> PathBuf {
        if let Ok(p) = std::env::var("OFAI_MAPS_DIR") {
            return PathBuf::from(p);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../OpenFrontIO/resources/maps")
    }

    fn load_bins(&mut self, map: &str) -> Result<&CachedBins, String> {
        let key = map.to_string();
        if !self.bins.contains_key(&key) {
            let dir = self.maps_dir.join(map_dir_name(map));
            let manifest: MapManifest = serde_json::from_str(
                &fs::read_to_string(dir.join("manifest.json"))
                    .map_err(|e| format!("read manifest {}: {e}", dir.display()))?,
            )
            .map_err(|e| format!("parse manifest: {e}"))?;
            let entry = CachedBins {
                manifest,
                map_bin: read_bin(&dir.join("map.bin"))?,
                map4x_bin: read_bin(&dir.join("map4x.bin"))?,
                map16x_bin: read_bin(&dir.join("map16x.bin"))?,
            };
            self.bins.insert(key.clone(), entry);
        }
        Ok(self.bins.get(&key).expect("just inserted"))
    }

    pub fn fresh_terrain(&mut self, map: &str, size: MapSize) -> Result<FreshTerrain, String> {
        let cached = self.load_bins(map)?;
        let compact = size == MapSize::Compact;
        let (meta, bin) = if compact {
            (&cached.manifest.map4x, &cached.map4x_bin)
        } else {
            (&cached.manifest.map, &cached.map_bin)
        };
        let (mini_meta, mini_bin) = if compact {
            (&cached.manifest.map16x, &cached.map16x_bin)
        } else {
            (&cached.manifest.map4x, &cached.map4x_bin)
        };
        let game_map = GameMap::new(meta.width, meta.height, bin.clone(), meta.num_land_tiles)?;
        let mini_map = GameMap::new(
            mini_meta.width,
            mini_meta.height,
            mini_bin.clone(),
            mini_meta.num_land_tiles,
        )?;
        let mut nations = cached.manifest.nations.clone();
        if compact {
            for n in &mut nations {
                if let Some([x, y]) = n.coordinates {
                    n.coordinates = Some([x / 2, y / 2]);
                }
            }
        }
        Ok(FreshTerrain {
            game_map,
            mini_map,
            nations,
            manifest: cached.manifest.clone(),
        })
    }
}

fn read_bin(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}

fn map_dir_name(map: &str) -> String {
    map.to_ascii_lowercase()
}
