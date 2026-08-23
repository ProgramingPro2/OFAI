//! Official-compatible water A* (mini-map), shore coercion, upscale, stepper.

use crate::map::{GameMap, TileRef};

const LAND_BIT: u8 = 7;
const MAGNITUDE_MASK: u8 = 0x1f;
const COST_SCALE: i32 = 100;
const BASE_COST: i32 = 1 * COST_SCALE;

fn magnitude_penalty(magnitude: u8) -> i32 {
    if magnitude < 3 {
        10 * COST_SCALE
    } else if magnitude <= 10 {
        0
    } else {
        COST_SCALE
    }
}

struct MinHeap {
    heap: Vec<u32>,
    pri: Vec<f32>,
}

impl MinHeap {
    fn new() -> Self {
        Self {
            heap: Vec::new(),
            pri: Vec::new(),
        }
    }
    fn clear(&mut self) {
        self.heap.clear();
        self.pri.clear();
    }
    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    fn push(&mut self, node: u32, priority: f32) {
        let mut i = self.heap.len();
        self.heap.push(node);
        self.pri.push(priority);
        while i > 0 {
            let parent = (i - 1) >> 1;
            if self.pri[parent] <= self.pri[i] {
                break;
            }
            self.heap.swap(i, parent);
            self.pri.swap(i, parent);
            i = parent;
        }
    }
    fn pop(&mut self) -> u32 {
        let result = self.heap[0];
        let last_n = self.heap.pop().unwrap();
        let last_p = self.pri.pop().unwrap();
        if self.heap.is_empty() {
            return result;
        }
        self.heap[0] = last_n;
        self.pri[0] = last_p;
        let mut i = 0;
        loop {
            let left = (i << 1) + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let mut smallest = i;
            if self.pri[left] < self.pri[smallest] {
                smallest = left;
            }
            if right < self.heap.len() && self.pri[right] < self.pri[smallest] {
                smallest = right;
            }
            if smallest == i {
                break;
            }
            self.heap.swap(i, smallest);
            self.pri.swap(i, smallest);
            i = smallest;
        }
        result
    }
}

/// Official AStarWater on a (usually mini) map.
pub struct AStarWater {
    stamp: u32,
    closed_stamp: Vec<u32>,
    g_stamp: Vec<u32>,
    g_score: Vec<u32>,
    came_from: Vec<i32>,
    queue: MinHeap,
    width: u32,
    num_nodes: usize,
    heuristic_weight: i32,
    max_iterations: i32,
}

impl AStarWater {
    pub fn new(map: &GameMap) -> Self {
        let n = map.len();
        Self {
            stamp: 1,
            closed_stamp: vec![0; n],
            g_stamp: vec![0; n],
            g_score: vec![0; n],
            came_from: vec![-1; n],
            queue: MinHeap::new(),
            width: map.width(),
            num_nodes: n,
            heuristic_weight: 5,
            max_iterations: 1_000_000,
        }
    }

    pub fn find_path(&mut self, map: &GameMap, start: &[TileRef], goal: TileRef) -> Option<Vec<TileRef>> {
        if start.is_empty() {
            return None;
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed_stamp.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let width = self.width;
        let num_nodes = self.num_nodes;
        let weight = self.heuristic_weight;
        let land_mask = 1u8 << LAND_BIT;
        let goal_x = (goal % width) as i32;
        let goal_y = (goal / width) as i32;

        self.queue.clear();
        let s0 = start[0];
        let start_x = (s0 % width) as i32;
        let start_y = (s0 / width) as i32;
        let dx_goal = goal_x - start_x;
        let dy_goal = goal_y - start_y;
        let cross_norm = 1.max(dx_goal.abs() + dy_goal.abs());

        let cross_tb = |nx: i32, ny: i32| -> i32 {
            let dx_n = nx - goal_x;
            let dy_n = ny - goal_y;
            let cross = (dx_goal * dy_n - dy_goal * dx_n).abs();
            ((cross * (COST_SCALE - 1)) / cross_norm) / cross_norm
        };

        for &s in start {
            let si = s as usize;
            self.g_score[si] = 0;
            self.g_stamp[si] = stamp;
            self.came_from[si] = -1;
            let sx = (s % width) as i32;
            let sy = (s / width) as i32;
            let h = weight * BASE_COST * ((sx - goal_x).abs() + (sy - goal_y).abs());
            self.queue.push(s, h as f32);
        }

        let mut iterations = self.max_iterations;
        while !self.queue.is_empty() {
            iterations -= 1;
            if iterations <= 0 {
                return None;
            }
            let current = self.queue.pop();
            let ci = current as usize;
            if self.closed_stamp[ci] == stamp {
                continue;
            }
            self.closed_stamp[ci] = stamp;
            if current == goal {
                return Some(self.build_path(goal));
            }
            let current_g = self.g_score[ci];
            let current_x = (current % width) as i32;
            let current_y = (current / width) as i32;

            let mut try_nb = |neighbor: u32, nx: i32, ny: i32| {
                let ni = neighbor as usize;
                if self.closed_stamp[ni] == stamp {
                    return;
                }
                let terrain = map.terrain_byte(neighbor);
                if neighbor != goal && (terrain & land_mask) != 0 {
                    return;
                }
                let mag = terrain & MAGNITUDE_MASK;
                let cost = BASE_COST + magnitude_penalty(mag);
                let tentative = current_g + cost as u32;
                if self.g_stamp[ni] != stamp || tentative < self.g_score[ni] {
                    self.came_from[ni] = current as i32;
                    self.g_score[ni] = tentative;
                    self.g_stamp[ni] = stamp;
                    let h = weight * BASE_COST * ((nx - goal_x).abs() + (ny - goal_y).abs());
                    let f = tentative as i32 + h + cross_tb(nx, ny);
                    self.queue.push(neighbor, f as f32);
                }
            };

            if current >= width {
                try_nb(current - width, current_x, current_y - 1);
            }
            if (current as usize) < num_nodes - width as usize {
                try_nb(current + width, current_x, current_y + 1);
            }
            if current_x != 0 {
                try_nb(current - 1, current_x - 1, current_y);
            }
            if current_x != width as i32 - 1 {
                try_nb(current + 1, current_x + 1, current_y);
            }
        }
        None
    }

    fn build_path(&self, goal: TileRef) -> Vec<TileRef> {
        let mut path = Vec::new();
        let mut current = goal as i32;
        while current != -1 {
            path.push(current as TileRef);
            current = self.came_from[current as usize];
        }
        path.reverse();
        path
    }
}

fn best_water_neighbor(map: &GameMap, tile: TileRef) -> Option<TileRef> {
    let mut nbuf = [0u32; 4];
    let n = map.neighbors4(tile, &mut nbuf);
    let mut best = None;
    let mut max_score = -1i32;
    for i in 0..n {
        let nb = nbuf[i];
        if !map.is_water(nb) {
            continue;
        }
        let mut nbuf2 = [0u32; 4];
        let n2 = map.neighbors4(nb, &mut nbuf2);
        let mut score = 0i32;
        for j in 0..n2 {
            if map.is_water(nbuf2[j]) {
                score += 1;
            }
        }
        if score > max_score {
            max_score = score;
            best = Some(nb);
        }
    }
    best
}

/// Shore-coerce then A* on the same map, then restore shore extremes.
pub fn find_water_path_on(
    astar: &mut AStarWater,
    map: &GameMap,
    from: &[TileRef],
    to: TileRef,
) -> Option<Vec<TileRef>> {
    let mut water_from = Vec::new();
    let mut water_to_original: Vec<(TileRef, TileRef)> = Vec::new();
    for &f in from {
        if map.is_water(f) {
            water_from.push(f);
            water_to_original.retain(|(w, _)| *w != f);
        } else if let Some(w) = best_water_neighbor(map, f) {
            water_from.push(w);
            water_to_original.retain(|(x, _)| *x != w);
            water_to_original.push((w, f));
        }
    }
    if water_from.is_empty() {
        return None;
    }
    let (water_to, original_to) = if map.is_water(to) {
        (to, None)
    } else {
        let w = best_water_neighbor(map, to)?;
        (w, Some(to))
    };
    let mut path = astar.find_path(map, &water_from, water_to)?;
    if path.is_empty() {
        return None;
    }
    if let Some((_, shore)) = water_to_original.iter().find(|(w, _)| *w == path[0]) {
        path.insert(0, *shore);
    }
    if let Some(orig) = original_to {
        if path.last().copied() != Some(orig) {
            path.push(orig);
        }
    }
    Some(path)
}

fn upscale_path(mini: &GameMap, main: &GameMap, path: &[TileRef]) -> Vec<TileRef> {
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
            let x = (cur_x as i32 + (dx * step) / steps) as u32;
            let y = (cur_y as i32 + (dy * step) / steps) as u32;
            if let Ok(r) = main.ref_xy(x, y) {
                smooth.push(r);
            }
        }
    }
    let last = path[path.len() - 1];
    if let Ok(r) = main.ref_xy(mini.x(last) * 2, mini.y(last) * 2) {
        smooth.push(r);
    }
    smooth
}

fn fix_extremes(mut upscaled: Vec<TileRef>, dst: TileRef, src: Option<TileRef>) -> Vec<TileRef> {
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

/// Official MiniMapTransformer + ShoreCoercing on the mini-map, then upscale.
pub fn find_water_path(
    astar: &mut AStarWater,
    map: &GameMap,
    mini: &GameMap,
    from: &[TileRef],
    to: TileRef,
) -> Option<Vec<TileRef>> {
    let mini_from: Vec<TileRef> = from
        .iter()
        .filter_map(|&f| mini.ref_xy(map.x(f) / 2, map.y(f) / 2).ok())
        .collect();
    let mini_to = mini.ref_xy(map.x(to) / 2, map.y(to) / 2).ok()?;
    let path = find_water_path_on(astar, mini, &mini_from, mini_to)?;
    if path.is_empty() {
        return None;
    }
    let mut upscaled = upscale_path(mini, map, &path);
    let src = if from.len() == 1 {
        Some(from[0])
    } else if !upscaled.is_empty() {
        let sx = map.x(upscaled[0]);
        let sy = map.y(upscaled[0]);
        let mut best = from[0];
        let mut best_d = u32::MAX;
        for &f in from {
            let d = map.x(f).abs_diff(sx) + map.y(f).abs_diff(sy);
            if d < best_d {
                best_d = d;
                best = f;
            }
        }
        Some(best)
    } else {
        None
    };
    Some(fix_extremes(upscaled, to, src))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathStatus {
    Complete,
    Next,
    NotFound,
}

pub struct PathStepper {
    path: Option<Vec<TileRef>>,
    index: usize,
    last_to: Option<TileRef>,
}

impl PathStepper {
    pub fn new() -> Self {
        Self {
            path: None,
            index: 0,
            last_to: None,
        }
    }

    pub fn next(
        &mut self,
        chain: &mut crate::hpa::WaterChain,
        map: &GameMap,
        mini: &GameMap,
        from: TileRef,
        to: TileRef,
    ) -> (PathStatus, Option<TileRef>) {
        if from == to {
            return (PathStatus::Complete, Some(to));
        }
        if self.last_to != Some(to) {
            self.path = None;
            self.index = 0;
            self.last_to = Some(to);
        }
        if self.path.is_none() {
            let found = chain.find_path(map, mini, &[from], to);
            if found.is_none() {
                return (PathStatus::NotFound, None);
            }
            let path = found.unwrap();
            self.index = if !path.is_empty() && path[0] == from {
                1
            } else {
                0
            };
            self.path = Some(path);
        }
        let path = self.path.as_ref().unwrap();
        if self.index > 0 && path.get(self.index - 1) != Some(&from) {
            self.path = None;
            self.index = 0;
            self.last_to = Some(to);
            return self.next(chain, map, mini, from, to);
        }
        if self.index >= path.len() {
            return (PathStatus::Complete, Some(to));
        }
        let node = path[self.index];
        self.index += 1;
        (PathStatus::Next, Some(node))
    }
}

/// Official refineStartTile (CANDIDATE_RADIUS / waypoint binary search + bounded A*).
pub fn refine_start_tile(
    astar: &mut AStarWater,
    map: &GameMap,
    path: &[TileRef],
    shores: &[TileRef],
) -> TileRef {
    const CANDIDATE_RADIUS: u32 = 20;
    const MIN_WAYPOINT: usize = 50;
    const MAX_WAYPOINT: usize = 200;
    const PADDING: i32 = 10;
    const REFINE_MAX: i32 = 100 * 100;

    if path.len() <= MIN_WAYPOINT {
        return path[0];
    }
    let best_tile = path[0];
    let candidates: Vec<TileRef> = shores
        .iter()
        .copied()
        .filter(|&s| map.manhattan(s, best_tile) <= CANDIDATE_RADIUS)
        .collect();
    if candidates.len() <= 1 {
        return best_tile;
    }
    let mut cand_min_x = map.x(candidates[0]) as i32;
    let mut cand_max_x = cand_min_x;
    let mut cand_min_y = map.y(candidates[0]) as i32;
    let mut cand_max_y = cand_min_y;
    for &c in &candidates[1..] {
        let sx = map.x(c) as i32;
        let sy = map.y(c) as i32;
        cand_min_x = cand_min_x.min(sx);
        cand_max_x = cand_max_x.max(sx);
        cand_min_y = cand_min_y.min(sy);
        cand_max_y = cand_max_y.max(sy);
    }
    let mut lo = MIN_WAYPOINT;
    let mut hi = MAX_WAYPOINT.min(path.len() - 1);
    let mut best_wp = lo;
    for _ in 0..5 {
        if lo > hi {
            break;
        }
        let mid = (lo + hi) >> 1;
        let wp = path[mid];
        let wpx = map.x(wp) as i32;
        let wpy = map.y(wp) as i32;
        let min_x = cand_min_x.min(wpx) - PADDING;
        let max_x = cand_max_x.max(wpx) + PADDING;
        let min_y = cand_min_y.min(wpy) - PADDING;
        let max_y = cand_max_y.max(wpy) + PADDING;
        let area = (max_x - min_x + 1) * (max_y - min_y + 1);
        if area <= REFINE_MAX {
            best_wp = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    let waypoint = path[best_wp];
    let wpx = map.x(waypoint) as i32;
    let wpy = map.y(waypoint) as i32;
    let bounds = (
        0.max(cand_min_x.min(wpx) - PADDING) as u32,
        ((map.width() as i32 - 1).min(cand_max_x.max(wpx) + PADDING)) as u32,
        0.max(cand_min_y.min(wpy) - PADDING) as u32,
        ((map.height() as i32 - 1).min(cand_max_y.max(wpy) + PADDING)) as u32,
    );
    let area = (bounds.1 as i32 - bounds.0 as i32 + 1) * (bounds.3 as i32 - bounds.2 as i32 + 1);
    if area > REFINE_MAX {
        return best_tile;
    }
    astar
        .find_path(map, &candidates, waypoint)
        .and_then(|p| p.first().copied())
        .unwrap_or(best_tile)
}
