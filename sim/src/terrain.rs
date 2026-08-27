//! Training-safe terrain loader: cache immutable bins, water, and HPA graphs.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::hpa::{build_abstract_graph, AbstractGraph};
use crate::map::{GameMap, WaterComponents};
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
    #[serde(default, rename = "additionalNations")]
    pub additional_nations: Vec<ManifestNation>,
}

#[derive(Clone)]
struct CachedBins {
    manifest: MapManifest,
    map_bin: Arc<[u8]>,
    map4x_bin: Arc<[u8]>,
    map16x_bin: Arc<[u8]>,
}

#[derive(Clone)]
struct CachedRuntime {
    water: WaterComponents,
    mini_water: WaterComponents,
    hpa_graph: Option<Arc<AbstractGraph>>,
}

/// Shared immutable map assets so each env can reset without a global cache lock.
#[derive(Clone)]
pub struct MapAssets {
    pub game_terrain: Arc<[u8]>,
    pub mini_terrain: Arc<[u8]>,
    pub game_w: u32,
    pub game_h: u32,
    pub game_land: u32,
    pub mini_w: u32,
    pub mini_h: u32,
    pub mini_land: u32,
    pub water: WaterComponents,
    pub mini_water: WaterComponents,
    pub hpa_graph: Option<Arc<AbstractGraph>>,
    pub nations: Vec<ManifestNation>,
    pub additional_nations: Vec<ManifestNation>,
}

impl MapAssets {
    pub fn fresh_maps(&self) -> Result<(GameMap, GameMap), String> {
        let game_map = GameMap::from_shared(
            self.game_w,
            self.game_h,
            Arc::clone(&self.game_terrain),
            self.game_land,
        )?;
        let mini_map = GameMap::from_shared(
            self.mini_w,
            self.mini_h,
            Arc::clone(&self.mini_terrain),
            self.mini_land,
        )?;
        Ok((game_map, mini_map))
    }
}

pub struct FreshTerrain {
    pub game_map: GameMap,
    pub mini_map: GameMap,
    pub manifest: MapManifest,
    pub nations: Vec<ManifestNation>,
    pub additional_nations: Vec<ManifestNation>,
    pub assets: MapAssets,
}

#[derive(Clone)]
pub struct TerrainCache {
    maps_dir: PathBuf,
    bins: HashMap<String, CachedBins>,
    runtime: HashMap<(String, bool), CachedRuntime>,
}

impl TerrainCache {
    pub fn new(maps_dir: impl Into<PathBuf>) -> Self {
        Self {
            maps_dir: maps_dir.into(),
            bins: HashMap::new(),
            runtime: HashMap::new(),
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
                map_bin: Arc::from(read_bin(&dir.join("map.bin"))?),
                map4x_bin: Arc::from(read_bin(&dir.join("map4x.bin"))?),
                map16x_bin: Arc::from(read_bin(&dir.join("map16x.bin"))?),
            };
            self.bins.insert(key.clone(), entry);
        }
        Ok(self.bins.get(&key).expect("just inserted"))
    }

    pub fn fresh_terrain(&mut self, map: &str, size: MapSize) -> Result<FreshTerrain, String> {
        let compact = size == MapSize::Compact;
        let rt_key = (map.to_string(), compact);
        self.load_bins(map)?;
        let cached = self.bins.get(map).expect("loaded");
        let (meta, bin) = if compact {
            (&cached.manifest.map4x, Arc::clone(&cached.map4x_bin))
        } else {
            (&cached.manifest.map, Arc::clone(&cached.map_bin))
        };
        let (mini_meta, mini_bin) = if compact {
            (&cached.manifest.map16x, Arc::clone(&cached.map16x_bin))
        } else {
            (&cached.manifest.map4x, Arc::clone(&cached.map4x_bin))
        };
        let game_w = meta.width;
        let game_h = meta.height;
        let game_land = meta.num_land_tiles;
        let mini_w = mini_meta.width;
        let mini_h = mini_meta.height;
        let mini_land = mini_meta.num_land_tiles;
        let mut nations = cached.manifest.nations.clone();
        let mut additional_nations = cached.manifest.additional_nations.clone();
        let manifest = cached.manifest.clone();
        if compact {
            for n in nations.iter_mut().chain(additional_nations.iter_mut()) {
                if let Some([x, y]) = n.coordinates {
                    n.coordinates = Some([x / 2, y / 2]);
                }
            }
        }
        if !self.runtime.contains_key(&rt_key) {
            let game_tmp = GameMap::from_shared(game_w, game_h, Arc::clone(&bin), game_land)?;
            let mini_tmp = GameMap::from_shared(mini_w, mini_h, Arc::clone(&mini_bin), mini_land)?;
            let water = WaterComponents::compute(&game_tmp);
            let mini_water = WaterComponents::compute(&mini_tmp);
            let graph = Arc::new(build_abstract_graph(&mini_tmp));
            let hpa_graph = if graph.node_count() >= 100 {
                Some(graph)
            } else {
                None
            };
            self.runtime.insert(
                rt_key.clone(),
                CachedRuntime {
                    water,
                    mini_water,
                    hpa_graph,
                },
            );
        }
        let rt = self.runtime.get(&rt_key).expect("runtime");
        let assets = MapAssets {
            game_terrain: Arc::clone(&bin),
            mini_terrain: Arc::clone(&mini_bin),
            game_w,
            game_h,
            game_land,
            mini_w,
            mini_h,
            mini_land,
            water: rt.water.clone(),
            mini_water: rt.mini_water.clone(),
            hpa_graph: rt.hpa_graph.clone(),
            nations: nations.clone(),
            additional_nations: additional_nations.clone(),
        };
        let (game_map, mini_map) = assets.fresh_maps()?;
        Ok(FreshTerrain {
            game_map,
            mini_map,
            nations,
            additional_nations,
            manifest,
            assets,
        })
    }
}

fn read_bin(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}

fn map_dir_name(map: &str) -> String {
    map.to_ascii_lowercase()
}
