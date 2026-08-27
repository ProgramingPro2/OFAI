//! Tribe names and nation selection matching official spawn order.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::game::NationSpec;
use crate::rng::PseudoRandom;
use crate::spec::{MapSize, NationsSpec};
use crate::terrain::ManifestNation;

#[derive(Deserialize)]
struct Theme {
    prefixes: Vec<String>,
    suffixes: Vec<String>,
}

fn themes() -> &'static HashMap<String, Theme> {
    static THEMES: OnceLock<HashMap<String, Theme>> = OnceLock::new();
    THEMES.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../OpenFrontIO/resources/tribeNameThemes.json");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|_| "{}".into());
        serde_json::from_str(&raw).unwrap_or_default()
    })
}

fn theme_names_for_map(map: &str) -> Vec<&'static str> {
    match map {
        "Halkidiki" => vec!["europe"],
        "FourIslands" => vec!["default"],
        _ => vec!["default"],
    }
}

pub fn create_nations(
    spec: &NationsSpec,
    manifest: &[ManifestNation],
    additional: &[ManifestNation],
    map_size: MapSize,
    rng: &mut PseudoRandom,
) -> Vec<NationSpec> {
    let _ = map_size;
    let to_nation = |n: &ManifestNation, rng: &mut PseudoRandom| -> NationSpec {
        let (sx, sy) = match n.coordinates {
            Some([x, y]) => (Some(x), Some(y)),
            None => (None, None),
        };
        NationSpec {
            name: n.name.clone(),
            spawn_x: sx,
            spawn_y: sy,
            id: rng.next_id(),
        }
    };
    match spec {
        NationsSpec::Disabled => Vec::new(),
        NationsSpec::Count(n) => {
            create_random_nations(*n as usize, manifest, additional, to_nation, rng)
        }
        NationsSpec::Default => manifest.iter().map(|m| to_nation(m, rng)).collect(),
    }
}

fn create_random_nations(
    target: usize,
    manifest: &[ManifestNation],
    additional: &[ManifestNation],
    to_nation: impl Fn(&ManifestNation, &mut PseudoRandom) -> NationSpec,
    rng: &mut PseudoRandom,
) -> Vec<NationSpec> {
    let shuffled = rng.shuffle_array(manifest);
    if target <= manifest.len() {
        return shuffled
            .iter()
            .take(target)
            .map(|m| to_nation(m, rng))
            .collect();
    }
    let mut nations: Vec<NationSpec> = shuffled.iter().map(|m| to_nation(m, rng)).collect();
    let mut used: std::collections::HashSet<String> =
        nations.iter().map(|n| n.name.clone()).collect();
    let mut remaining = target - manifest.len();
    if remaining > 0 && !additional.is_empty() {
        let candidates: Vec<ManifestNation> = additional
            .iter()
            .filter(|n| !used.contains(&n.name))
            .cloned()
            .collect();
        let extras = rng.shuffle_array(&candidates);
        for extra in extras.into_iter().take(remaining) {
            used.insert(extra.name.clone());
            nations.push(to_nation(&extra, rng));
            remaining -= 1;
        }
    }
    for _ in 0..remaining {
        let name = crate::nation_names::generate_unique_nation_name(rng, &used);
        used.insert(name.clone());
        nations.push(NationSpec {
            name,
            spawn_x: None,
            spawn_y: None,
            id: rng.next_id(),
        });
    }
    nations
}

pub fn tribe_name(map: &str, rng: &mut PseudoRandom) -> String {
    let mut prefixes = Vec::new();
    let mut suffixes = Vec::new();
    for name in theme_names_for_map(map) {
        if let Some(t) = themes().get(name) {
            prefixes.extend(t.prefixes.iter().cloned());
            suffixes.extend(t.suffixes.iter().cloned());
        }
    }
    if prefixes.is_empty() || suffixes.is_empty() {
        if let Some(t) = themes().get("default") {
            prefixes = t.prefixes.clone();
            suffixes = t.suffixes.clone();
        }
    }
    if prefixes.is_empty() {
        prefixes.push("Tribe".into());
    }
    if suffixes.is_empty() {
        suffixes.push("Folk".into());
    }
    let p = rng.next_int(0.0, prefixes.len() as f64) as usize;
    let s = rng.next_int(0.0, suffixes.len() as f64) as usize;
    format!(
        "{} {}",
        prefixes[p.min(prefixes.len() - 1)],
        suffixes[s.min(suffixes.len() - 1)]
    )
}
