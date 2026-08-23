//! Official SpatialQuery dest / source selection.

use std::collections::HashSet;

use crate::map::{GameMap, TileRef, WaterComponents};
use crate::hpa::{new_refine_astar, refine_start_tile_bounded, WaterChain};

/// Official WaterManager.getWaterComponent via mini-map components.
pub fn get_water_component(
    map: &GameMap,
    mini: &GameMap,
    mini_water: &WaterComponents,
    tile: TileRef,
) -> Option<i32> {
    let mx = map.x(tile) / 2;
    let my = map.y(tile) / 2;
    let Ok(mini_tile) = mini.ref_xy(mx, my) else {
        return None;
    };
    if mini.is_water(mini_tile) {
        return mini_water.of_tile(mini, mini_tile);
    }
    let mut nbuf = [0u32; 4];
    let n = mini.neighbors4(mini_tile, &mut nbuf);
    for i in 0..n {
        if mini.is_water(nbuf[i]) {
            if let Some(id) = mini_water.of_tile(mini, nbuf[i]) {
                return Some(id);
            }
        }
    }
    for i in 0..n {
        let mut nbuf2 = [0u32; 4];
        let n2 = mini.neighbors4(nbuf[i], &mut nbuf2);
        for j in 0..n2 {
            if mini.is_water(nbuf2[j]) {
                if let Some(id) = mini_water.of_tile(mini, nbuf2[j]) {
                    return Some(id);
                }
            }
        }
    }
    None
}

/// Official `WaterManager.hasWaterComponent`.
pub fn has_water_component(
    map: &GameMap,
    mini: &GameMap,
    mini_water: &WaterComponents,
    tile: TileRef,
    component: i32,
) -> bool {
    let mx = map.x(tile) / 2;
    let my = map.y(tile) / 2;
    let Ok(mini_tile) = mini.ref_xy(mx, my) else {
        return false;
    };
    if mini.is_water(mini_tile) && mini_water.of_tile(mini, mini_tile) == Some(component) {
        return true;
    }
    let mut nbuf = [0u32; 4];
    let n = mini.neighbors4(mini_tile, &mut nbuf);
    for i in 0..n {
        if mini.is_water(nbuf[i]) && mini_water.of_tile(mini, nbuf[i]) == Some(component) {
            return true;
        }
    }
    for i in 0..n {
        let mut nbuf2 = [0u32; 4];
        let n2 = mini.neighbors4(nbuf[i], &mut nbuf2);
        for j in 0..n2 {
            if mini.is_water(nbuf2[j]) && mini_water.of_tile(mini, nbuf2[j]) == Some(component) {
                return true;
            }
        }
    }
    false
}

fn bfs_nearest(
    map: &GameMap,
    from: TileRef,
    max_dist: u32,
    mut pred: impl FnMut(TileRef) -> bool,
) -> Option<TileRef> {
    if !map.is_valid_ref(from as i32) {
        return None;
    }
    let mut seen = vec![false; map.len()];
    let mut stack = Vec::new();
    let mut best = None;
    let mut best_dist = u32::MAX;
    let mut mark = |t: TileRef, seen: &mut [bool], stack: &mut Vec<TileRef>| {
        seen[t as usize] = true;
        stack.push(t);
    };
    if max_dist as i32 >= 0 {
        mark(from, &mut seen, &mut stack);
        if pred(from) {
            best = Some(from);
            best_dist = 0;
        }
    }
    let mut nbuf = [0u32; 4];
    while let Some(curr) = stack.pop() {
        let n = map.neighbors4(curr, &mut nbuf);
        for i in 0..n {
            let nb = nbuf[i];
            if seen[nb as usize] {
                continue;
            }
            if map.manhattan(from, nb) > max_dist {
                continue;
            }
            seen[nb as usize] = true;
            stack.push(nb);
            if pred(nb) {
                let d = map.manhattan(from, nb);
                if d < best_dist {
                    best_dist = d;
                    best = Some(nb);
                }
            }
        }
    }
    best
}

pub fn closest_reachable_shore(
    map: &GameMap,
    mini: &GameMap,
    mini_water: &WaterComponents,
    target_sid: u16,
    attacker_shores: impl Iterator<Item = TileRef>,
    tile: TileRef,
    max_dist: u32,
) -> Option<TileRef> {
    let mut reachable = HashSet::new();
    for t in attacker_shores {
        if !map.is_shore(t) || !map.is_land(t) {
            continue;
        }
        if let Some(c) = get_water_component(map, mini, mini_water, t) {
            reachable.insert(c);
        }
    }
    if reachable.is_empty() {
        return None;
    }
    bfs_nearest(map, tile, max_dist, |t| {
        if !map.is_shore(t) || !map.is_land(t) {
            return false;
        }
        if map.owner_id(t) != target_sid {
            return false;
        }
        get_water_component(map, mini, mini_water, t)
            .map(|c| reachable.contains(&c))
            .unwrap_or(false)
    })
}

pub fn closest_shore_by_water(
    chain: &mut WaterChain,
    map: &GameMap,
    mini: &GameMap,
    mini_water: &WaterComponents,
    owner_shores: &[TileRef],
    target: TileRef,
) -> Option<TileRef> {
    if !map.is_water(target) && !map.is_shore(target) {
        return None;
    }
    let target_c = get_water_component(map, mini, mini_water, target)?;
    let shores: Vec<TileRef> = owner_shores
        .iter()
        .copied()
        .filter(|&t| {
            map.is_shore(t)
                && map.is_land(t)
                && get_water_component(map, mini, mini_water, t) == Some(target_c)
        })
        .collect();
    if shores.is_empty() {
        return None;
    }
    let path = chain.find_path(map, mini, &shores, target)?;
    if path.is_empty() {
        return None;
    }
    let mut bounded = new_refine_astar(map);
    Some(refine_start_tile_bounded(&mut bounded, map, &path, &shores))
}

pub fn target_transport_tile(
    map: &GameMap,
    mini: &GameMap,
    mini_water: &WaterComponents,
    target_sid: u16,
    attacker_border: impl Iterator<Item = TileRef>,
    tile: TileRef,
) -> Option<TileRef> {
    closest_reachable_shore(map, mini, mini_water, target_sid, attacker_border, tile, 50)
}

pub fn can_build_transport_ship(
    chain: &mut WaterChain,
    map: &GameMap,
    mini: &GameMap,
    mini_water: &WaterComponents,
    boat_count: u32,
    owner_small: u16,
    owner_shores: &[TileRef],
    owner_border: impl Iterator<Item = TileRef>,
    tile: TileRef,
    can_attack_owner: impl FnOnce(u16) -> bool,
) -> Option<TileRef> {
    if boat_count >= crate::units::BOAT_MAX_NUMBER {
        return None;
    }
    let dst = target_transport_tile(map, mini, mini_water, map.owner_id(tile), owner_border, tile)?;
    if map.owner_id(tile) == owner_small {
        return None;
    }
    let other = map.owner_id(tile);
    if other != 0 && !can_attack_owner(other) {
        return None;
    }
    closest_shore_by_water(chain, map, mini, mini_water, owner_shores, dst)
}
