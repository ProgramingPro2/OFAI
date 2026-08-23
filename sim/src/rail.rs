//! Official rail A*, station graph, clusters, and railroads.

use std::collections::{HashMap, HashSet, VecDeque};

use indexmap::IndexSet;

use crate::config::{TRAIN_STATION_MAX_RANGE, TRAIN_STATION_MIN_RANGE};
use crate::map::{GameMap, TileRef};
use crate::units::UnitType;

const WATER_PENALTY: i32 = 5;
const HEURISTIC_WEIGHT: i32 = 2;
const DIR_CHANGE: i32 = 3;
const MAX_CONNECTION: i32 = 4;
const STATION_RADIUS: u32 = 3;

pub struct BucketQueue {
    buckets: Vec<Vec<u32>>,
    min_bucket: usize,
    size: usize,
}

impl BucketQueue {
    pub fn new(max_priority: usize) -> Self {
        Self {
            buckets: vec![Vec::new(); max_priority.max(1)],
            min_bucket: max_priority.max(1),
            size: 0,
        }
    }

    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.min_bucket = self.buckets.len();
        self.size = 0;
    }

    pub fn push(&mut self, node: u32, priority: i32) {
        let bucket = (priority.max(0) as usize).min(self.buckets.len() - 1);
        self.buckets[bucket].push(node);
        self.size += 1;
        if bucket < self.min_bucket {
            self.min_bucket = bucket;
        }
    }

    pub fn pop(&mut self) -> u32 {
        while self.min_bucket < self.buckets.len() {
            if let Some(n) = self.buckets[self.min_bucket].pop() {
                self.size -= 1;
                return n;
            }
            self.min_bucket += 1;
        }
        u32::MAX
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

fn js_round(x: f64) -> i32 {
    // Official Math.round: half toward +inf.
    (x + 0.5).floor() as i32
}

/// Official PathFinding.Rail: AStarRail on miniMap, then MiniMapTransformer.
pub fn find_rail_path(map: &GameMap, mini: &GameMap, from: TileRef, to: TileRef) -> Vec<TileRef> {
    let Ok(mini_from) = mini.ref_xy(map.x(from) / 2, map.y(from) / 2) else {
        return Vec::new();
    };
    let Ok(mini_to) = mini.ref_xy(map.x(to) / 2, map.y(to) / 2) else {
        return Vec::new();
    };
    let mini_path = astar_rail(mini, mini_from, mini_to);
    if mini_path.is_empty() {
        return Vec::new();
    }
    fix_rail_extremes(upscale_rail_path(mini, map, &mini_path), to, Some(from))
}

fn astar_rail(map: &GameMap, from: TileRef, to: TileRef) -> Vec<TileRef> {
    let w = map.width();
    let h = map.height();
    let n = (w * h) as usize;
    let max_pri = (HEURISTIC_WEIGHT * (w + h) as i32 * (1 + WATER_PENALTY + DIR_CHANGE)) as usize + 1;
    let mut q = BucketQueue::new(max_pri);
    let mut closed = vec![0u32; n];
    let mut g_stamp = vec![0u32; n];
    let mut g = vec![0u32; n];
    let mut came = vec![-1i32; n];
    let stamp = 1u32;
    g[from as usize] = 0;
    g_stamp[from as usize] = stamp;
    came[from as usize] = -1;
    q.push(from, rail_h(map, from, to));
    let mut iters = 500_000i32;
    let mut nbuf = [0u32; 4];
    while !q.is_empty() {
        iters -= 1;
        if iters <= 0 {
            return Vec::new();
        }
        let current = q.pop();
        if closed[current as usize] == stamp {
            continue;
        }
        closed[current as usize] = stamp;
        if current == to {
            let mut path = Vec::new();
            let mut c = current as i32;
            while c != -1 {
                path.push(c as TileRef);
                c = came[c as usize];
            }
            path.reverse();
            return path;
        }
        let current_g = g[current as usize];
        let prev = came[current as usize];
        let from_shore = map.is_shoreline(current);
        let count = rail_neighbors(map, current, from_shore, &mut nbuf);
        for i in 0..count {
            let nb = nbuf[i];
            if closed[nb as usize] == stamp {
                continue;
            }
            let cost = rail_cost(map, current, nb, if prev == -1 { None } else { Some(prev as TileRef) });
            let tentative = current_g + cost as u32;
            if g_stamp[nb as usize] != stamp || tentative < g[nb as usize] {
                came[nb as usize] = current as i32;
                g[nb as usize] = tentative;
                g_stamp[nb as usize] = stamp;
                q.push(nb, tentative as i32 + rail_h(map, nb, to));
            }
        }
    }
    Vec::new()
}

fn upscale_rail_path(mini: &GameMap, main: &GameMap, path: &[TileRef]) -> Vec<TileRef> {
    let mut smooth = Vec::new();
    if path.is_empty() {
        return smooth;
    }
    for i in 0..path.len().saturating_sub(1) {
        let cur_x = mini.x(path[i]) * 2;
        let cur_y = mini.y(path[i]) * 2;
        let next_x = mini.x(path[i + 1]) * 2;
        let next_y = mini.y(path[i + 1]) * 2;
        if let Ok(r) = main.ref_xy(cur_x, cur_y) {
            smooth.push(r);
        }
        let dx = next_x as i32 - cur_x as i32;
        let dy = next_y as i32 - cur_y as i32;
        let steps = dx.abs().max(dy.abs());
        for step in 1..steps {
            let x = js_round(cur_x as f64 + (dx as f64 * step as f64) / steps as f64);
            let y = js_round(cur_y as f64 + (dy as f64 * step as f64) / steps as f64);
            if x >= 0 && y >= 0 {
                if let Ok(r) = main.ref_xy(x as u32, y as u32) {
                    smooth.push(r);
                }
            }
        }
    }
    let last = path[path.len() - 1];
    if let Ok(r) = main.ref_xy(mini.x(last) * 2, mini.y(last) * 2) {
        smooth.push(r);
    }
    smooth
}

fn fix_rail_extremes(mut upscaled: Vec<TileRef>, dst: TileRef, src: Option<TileRef>) -> Vec<TileRef> {
    if let Some(src) = src {
        if let Some(idx) = upscaled.iter().position(|&t| t == src) {
            if idx != 0 {
                upscaled = upscaled.split_off(idx);
            }
        } else {
            upscaled.insert(0, src);
        }
    }
    if let Some(idx) = upscaled.iter().position(|&t| t == dst) {
        if idx + 1 != upscaled.len() {
            upscaled.truncate(idx + 1);
        }
    } else {
        upscaled.push(dst);
    }
    upscaled
}

fn rail_neighbors(map: &GameMap, node: TileRef, from_shore: bool, buf: &mut [TileRef; 4]) -> usize {
    let mut n = 0;
    let mut nbuf = [0u32; 4];
    let c = map.neighbors4(node, &mut nbuf);
    for i in 0..c {
        let t = nbuf[i];
        if rail_traversable(map, t, from_shore) {
            buf[n] = t;
            n += 1;
        }
    }
    n
}

fn rail_traversable(map: &GameMap, to: TileRef, from_shore: bool) -> bool {
    if !map.is_water(to) {
        return true;
    }
    from_shore || map.is_shoreline(to)
}

fn rail_cost(map: &GameMap, from: TileRef, to: TileRef, prev: Option<TileRef>) -> i32 {
    let penalized = map.is_water(to) || map.is_shoreline(to);
    let mut c = if penalized { 1 + WATER_PENALTY } else { 1 };
    if let Some(p) = prev {
        let d1 = from as i32 - p as i32;
        let d2 = to as i32 - from as i32;
        if d1 != d2 {
            c += DIR_CHANGE;
        }
    }
    c
}

fn rail_h(map: &GameMap, node: TileRef, goal: TileRef) -> i32 {
    HEURISTIC_WEIGHT
        * ((map.x(node) as i32 - map.x(goal) as i32).abs()
            + (map.y(node) as i32 - map.y(goal) as i32).abs())
}

#[derive(Clone)]
pub struct Railroad {
    pub id: u32,
    pub from: u32,
    pub to: u32,
    pub tiles: Vec<TileRef>,
}

#[derive(Clone)]
pub struct Station {
    pub id: u32,
    pub unit_id: u32,
    pub tile: TileRef,
    pub cluster: Option<u32>,
    pub neighbors: Vec<u32>,
    pub rails: Vec<u32>,
}

pub struct Cluster {
    pub id: u32,
    pub stations: HashSet<u32>,
    pub trade: IndexSet<u32>,
}

pub struct RailNetwork {
    pub stations: HashMap<u32, Station>,
    pub by_unit: HashMap<u32, u32>,
    pub rails: HashMap<u32, Railroad>,
    pub clusters: HashMap<u32, Cluster>,
    next_station: u32,
    next_rail: u32,
    next_cluster: u32,
    dirty: HashSet<u32>,
}

impl RailNetwork {
    pub fn new() -> Self {
        Self {
            stations: HashMap::new(),
            by_unit: HashMap::new(),
            rails: HashMap::new(),
            clusters: HashMap::new(),
            next_station: 1,
            next_rail: 0,
            next_cluster: 1,
            dirty: HashSet::new(),
        }
    }

    pub fn find_by_unit(&self, unit_id: u32) -> Option<u32> {
        self.by_unit.get(&unit_id).copied()
    }

    pub fn connect_station(
        &mut self,
        map: &GameMap,
        mini: &GameMap,
        unit_id: u32,
        tile: TileRef,
        nearby: &[(u32, TileRef, i64)],
    ) -> u32 {
        let id = self.next_station;
        self.next_station += 1;
        self.stations.insert(
            id,
            Station {
                id,
                unit_id,
                tile,
                cluster: None,
                neighbors: Vec::new(),
                rails: Vec::new(),
            },
        );
        self.by_unit.insert(unit_id, id);
        if !self.snap_existing(map, id, tile) {
            self.connect_nearby(map, mini, id, nearby);
        }
        id
    }

    fn snap_existing(&mut self, map: &GameMap, sid: u32, tile: TileRef) -> bool {
        let rail_ids: Vec<u32> = self.rails.keys().copied().collect();
        let mut edited = HashSet::new();
        for rid in rail_ids {
            let Some(rail) = self.rails.get(&rid).cloned() else {
                continue;
            };
            if rail.tiles.iter().all(|&t| {
                let dx = map.x(t) as i64 - map.x(tile) as i64;
                let dy = map.y(t) as i64 - map.y(tile) as i64;
                dx * dx + dy * dy > (STATION_RADIUS as i64) * (STATION_RADIUS as i64)
            }) {
                continue;
            }
            let closest = closest_index(map, &rail.tiles, tile);
            if closest == 0 || closest >= rail.tiles.len() {
                continue;
            }
            self.remove_rail_links(rid);
            let left = rail.tiles[..closest].to_vec();
            let right = rail.tiles[closest..].to_vec();
            self.add_rail(rail.from, sid, left);
            self.add_rail(sid, rail.to, right);
            if let Some(c) = self.stations.get(&rail.from).and_then(|s| s.cluster) {
                self.add_to_cluster(c, sid);
                edited.insert(c);
            }
        }
        if edited.len() > 1 {
            self.merge_clusters(&edited);
        }
        !edited.is_empty()
    }

    fn connect_nearby(
        &mut self,
        map: &GameMap,
        mini: &GameMap,
        sid: u32,
        nearby: &[(u32, TileRef, i64)],
    ) {
        let mut edited = HashSet::new();
        let mut sorted = nearby.to_vec();
        sorted.sort_by_key(|x| x.2);
        let min2 = (TRAIN_STATION_MIN_RANGE as i64) * (TRAIN_STATION_MIN_RANGE as i64);
        for &(uid, _tile, d2) in &sorted {
            let Some(nid) = self.by_unit.get(&uid).copied() else {
                continue;
            };
            if nid == sid {
                continue;
            }
            let dist = self.graph_distance(nid, sid, MAX_CONNECTION);
            let Some(ncluster) = self.stations.get(&nid).and_then(|s| s.cluster) else {
                continue;
            };
            let available = dist > MAX_CONNECTION || dist == -1;
            if available && d2 > min2 {
                if self.connect_pair(map, mini, sid, nid) {
                    self.add_to_cluster(ncluster, sid);
                    edited.insert(ncluster);
                }
            }
        }
        if edited.len() > 1 {
            self.merge_clusters(&edited);
        } else if edited.is_empty() {
            let cid = self.next_cluster;
            self.next_cluster += 1;
            self.clusters.insert(
                cid,
                Cluster {
                    id: cid,
                    stations: HashSet::new(),
                    trade: IndexSet::new(),
                },
            );
            self.add_to_cluster(cid, sid);
        }
    }

    fn connect_pair(&mut self, map: &GameMap, mini: &GameMap, from: u32, to: u32) -> bool {
        let (a, b) = {
            let fa = self.stations.get(&from).map(|s| s.tile);
            let tb = self.stations.get(&to).map(|s| s.tile);
            match (fa, tb) {
                (Some(a), Some(b)) => (a, b),
                _ => return false,
            }
        };
        let path = find_rail_path(map, mini, a, b);
        let max = (TRAIN_STATION_MAX_RANGE as f64 * 1.4142) as usize;
        if path.is_empty() || path.len() >= max {
            return false;
        }
        self.add_rail(from, to, path);
        true
    }

    fn add_rail(&mut self, from: u32, to: u32, tiles: Vec<TileRef>) {
        let id = self.next_rail;
        self.next_rail += 1;
        self.rails.insert(
            id,
            Railroad {
                id,
                from,
                to,
                tiles,
            },
        );
        if let Some(s) = self.stations.get_mut(&from) {
            s.rails.push(id);
            s.neighbors.push(to);
        }
        if let Some(s) = self.stations.get_mut(&to) {
            s.rails.push(id);
            s.neighbors.push(from);
        }
    }

    fn remove_rail_links(&mut self, rid: u32) {
        if let Some(rail) = self.rails.remove(&rid) {
            if let Some(s) = self.stations.get_mut(&rail.from) {
                s.rails.retain(|x| *x != rid);
                s.neighbors.retain(|x| *x != rail.to);
            }
            if let Some(s) = self.stations.get_mut(&rail.to) {
                s.rails.retain(|x| *x != rid);
                s.neighbors.retain(|x| *x != rail.from);
            }
        }
    }

    fn add_to_cluster(&mut self, cid: u32, sid: u32) {
        if let Some(old) = self.stations.get(&sid).and_then(|s| s.cluster) {
            if old != cid {
                if let Some(c) = self.clusters.get_mut(&old) {
                    c.stations.remove(&sid);
                    c.trade.remove(&sid);
                }
            }
        }
        if let Some(s) = self.stations.get_mut(&sid) {
            s.cluster = Some(cid);
        }
        if let Some(c) = self.clusters.get_mut(&cid) {
            c.stations.insert(sid);
        }
    }

    fn merge_clusters(&mut self, ids: &HashSet<u32>) {
        let nid = self.next_cluster;
        self.next_cluster += 1;
        let mut stations = HashSet::new();
        for cid in ids {
            if let Some(c) = self.clusters.remove(cid) {
                stations.extend(c.stations);
            }
        }
        self.clusters.insert(
            nid,
            Cluster {
                id: nid,
                stations: HashSet::new(),
                trade: IndexSet::new(),
            },
        );
        for sid in stations {
            self.add_to_cluster(nid, sid);
        }
    }

    fn graph_distance(&self, start: u32, dest: u32, max_d: i32) -> i32 {
        if start == dest {
            return 0;
        }
        let mut visited = HashSet::new();
        let mut q = VecDeque::new();
        q.push_back((start, 0));
        while let Some((sid, d)) = q.pop_front() {
            if !visited.insert(sid) {
                continue;
            }
            if d >= max_d {
                continue;
            }
            if let Some(s) = self.stations.get(&sid) {
                for &n in &s.neighbors {
                    if n == dest {
                        return d + 1;
                    }
                    if !visited.contains(&n) {
                        q.push_back((n, d + 1));
                    }
                }
            }
        }
        -1
    }

    pub fn find_station_path(&self, map: &GameMap, from: u32, to: u32) -> Vec<u32> {
        let Some(fc) = self.stations.get(&from).and_then(|s| s.cluster) else {
            return Vec::new();
        };
        let Some(tc) = self.stations.get(&to).and_then(|s| s.cluster) else {
            return Vec::new();
        };
        if fc != tc {
            return Vec::new();
        }
        // Official StationPathFinder: A* with cost 1 and manhattan heuristic.
        let max_id = self.stations.keys().copied().max().unwrap_or(0) as usize + 1;
        let max_pri = (map.width() + map.height()) as usize + 1;
        let mut q = BucketQueue::new(max_pri.max(16));
        let mut closed = vec![0u32; max_id];
        let mut g_stamp = vec![0u32; max_id];
        let mut g = vec![0u32; max_id];
        let mut came = vec![-1i32; max_id];
        let stamp = 1u32;
        g[from as usize] = 0;
        g_stamp[from as usize] = stamp;
        came[from as usize] = -1;
        q.push(from, self.station_h(map, from, to));
        while !q.is_empty() {
            let current = q.pop();
            if current == u32::MAX || current as usize >= max_id {
                break;
            }
            if closed[current as usize] == stamp {
                continue;
            }
            closed[current as usize] = stamp;
            if current == to {
                let mut path = Vec::new();
                let mut c = current as i32;
                while c != -1 {
                    path.push(c as u32);
                    c = came[c as usize];
                }
                path.reverse();
                return path;
            }
            let current_g = g[current as usize];
            let Some(st) = self.stations.get(&current) else {
                continue;
            };
            for &nb in &st.neighbors {
                if nb as usize >= max_id || closed[nb as usize] == stamp {
                    continue;
                }
                let tentative = current_g + 1;
                if g_stamp[nb as usize] != stamp || tentative < g[nb as usize] {
                    came[nb as usize] = current as i32;
                    g[nb as usize] = tentative;
                    g_stamp[nb as usize] = stamp;
                    q.push(nb, tentative as i32 + self.station_h(map, nb, to));
                }
            }
        }
        Vec::new()
    }

    fn station_h(&self, map: &GameMap, a: u32, b: u32) -> i32 {
        let Some(sa) = self.stations.get(&a) else {
            return 0;
        };
        let Some(sb) = self.stations.get(&b) else {
            return 0;
        };
        (map.x(sa.tile) as i32 - map.x(sb.tile) as i32).abs()
            + (map.y(sa.tile) as i32 - map.y(sb.tile) as i32).abs()
    }

    pub fn oriented_tiles(&self, from: u32, to: u32) -> Option<Vec<TileRef>> {
        let s = self.stations.get(&from)?;
        for &rid in &s.rails {
            let r = self.rails.get(&rid)?;
            if r.from == from && r.to == to {
                return Some(r.tiles.clone());
            }
            if r.to == from && r.from == to {
                let mut t = r.tiles.clone();
                t.reverse();
                return Some(t);
            }
        }
        None
    }

    pub fn mark_trade(&mut self, sid: u32, is_trade: bool) {
        let Some(cid) = self.stations.get(&sid).and_then(|s| s.cluster) else {
            return;
        };
        if let Some(c) = self.clusters.get_mut(&cid) {
            if is_trade {
                c.trade.insert(sid);
            }
        }
    }

    pub fn trade_stations(&self, sid: u32) -> Vec<u32> {
        let Some(cid) = self.stations.get(&sid).and_then(|s| s.cluster) else {
            return Vec::new();
        };
        self.clusters
            .get(&cid)
            .map(|c| c.trade.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn recompute(&mut self) {
        let dirty: Vec<u32> = self.dirty.drain().collect();
        let _ = dirty;
    }

    pub fn remove_unit(&mut self, unit_id: u32) {
        let Some(sid) = self.by_unit.remove(&unit_id) else {
            return;
        };
        let rails: Vec<u32> = self
            .stations
            .get(&sid)
            .map(|s| s.rails.clone())
            .unwrap_or_default();
        for rid in rails {
            self.remove_rail_links(rid);
        }
        if let Some(s) = self.stations.remove(&sid) {
            if let Some(cid) = s.cluster {
                if let Some(c) = self.clusters.get_mut(&cid) {
                    c.stations.remove(&sid);
                    c.trade.remove(&sid);
                    self.dirty.insert(cid);
                }
            }
        }
    }
}

fn closest_index(map: &GameMap, tiles: &[TileRef], to: TileRef) -> usize {
    let tx = map.x(to) as i64;
    let ty = map.y(to) as i64;
    let mut best = 0usize;
    let mut best_d = i64::MAX;
    for (i, &t) in tiles.iter().enumerate() {
        let dx = map.x(t) as i64 - tx;
        let dy = map.y(t) as i64 - ty;
        let d = dx * dx + dy * dy;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

pub fn is_trade_type(ty: UnitType) -> bool {
    matches!(ty, UnitType::City | UnitType::Port)
}
