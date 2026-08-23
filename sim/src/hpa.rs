//! Official water HPA: AbstractGraph, hierarchical A*, bounded A*, smoothing.
//!
//! Mirrors OpenFront `src/core/pathfinding` so boat paths match the TypeScript
//! core when `graph.nodeCount >= 100`. Smaller graphs keep simple mini A*.

use crate::map::{GameMap, TileRef};

const LAND_BIT: u8 = 7;
const MAGNITUDE_MASK: u8 = 0x1f;
const COST_SCALE: i32 = 100;
const BASE_COST: i32 = COST_SCALE;
const CLUSTER_SIZE: u32 = 32;
const HPA_NODE_THRESHOLD: usize = 100;

const LOCAL_ASTAR_MAX_AREA: usize = 100 * 100;
const ENDPOINT_REFINEMENT_TILES: u32 = 50;
const LOS_MIN_MAGNITUDE_PASS1: u8 = 2;
const LOS_MIN_MAGNITUDE_PASS2: u8 = 3;
const SHORT_PATH_THRESHOLD: u32 = 120;
const SHORT_PATH_PADDING: i32 = 10;
const SHORT_PATH_SIZE: u32 = 260;

fn magnitude_penalty_simple(magnitude: u8) -> i32 {
    if magnitude < 3 {
        10 * COST_SCALE
    } else if magnitude <= 10 {
        0
    } else {
        COST_SCALE
    }
}

fn magnitude_penalty_bounded(magnitude: u8) -> i32 {
    if magnitude < 3 {
        3 * COST_SCALE
    } else if magnitude <= 10 {
        0
    } else {
        COST_SCALE
    }
}

fn js_round(x: f64) -> i32 {
    // JS Math.round: half away from -inf for positive, i.e. half toward +inf.
    (x + 0.5).floor() as i32
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
    fn with_capacity(n: usize) -> Self {
        Self {
            heap: Vec::with_capacity(n),
            pri: Vec::with_capacity(n),
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

struct BfsGrid {
    stamp: u32,
    visited: Vec<u32>,
    queue: Vec<i32>,
    dist: Vec<u16>,
}

impl BfsGrid {
    fn new(n: usize) -> Self {
        Self {
            stamp: 1,
            visited: vec![0; n],
            queue: vec![0; n],
            dist: vec![0; n],
        }
    }

    fn next_stamp(&mut self) -> u32 {
        let stamp = self.stamp;
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.visited.fill(0);
            self.stamp = 1;
        }
        stamp
    }

    fn search<R>(
        &mut self,
        width: u32,
        height: u32,
        start: &[TileRef],
        max_distance: u32,
        mut is_valid: impl FnMut(TileRef) -> bool,
        mut visitor: impl FnMut(TileRef, u32) -> BfsVisit<R>,
    ) -> Option<R> {
        let stamp = self.next_stamp();
        let last_row = (height.saturating_sub(1)) * width;
        let mut head = 0usize;
        let mut tail = 0usize;
        for &s in start {
            let si = s as usize;
            if si >= self.visited.len() {
                continue;
            }
            self.visited[si] = stamp;
            self.dist[si] = 0;
            self.queue[tail] = s as i32;
            tail += 1;
        }
        while head < tail {
            let node = self.queue[head] as TileRef;
            head += 1;
            let dist = self.dist[node as usize] as u32;
            match visitor(node, dist) {
                BfsVisit::Found(r) => return Some(r),
                BfsVisit::Reject => continue,
                BfsVisit::Continue => {}
            }
            let next_dist = dist + 1;
            if next_dist > max_distance {
                continue;
            }
            let x = node % width;
            let mut try_nb = |this: &mut Self, n: TileRef, tail: &mut usize| {
                let ni = n as usize;
                if this.visited[ni] != stamp && is_valid(n) {
                    this.visited[ni] = stamp;
                    this.dist[ni] = next_dist as u16;
                    this.queue[*tail] = n as i32;
                    *tail += 1;
                }
            };
            if node >= width {
                try_nb(self, node - width, &mut tail);
            }
            if node < last_row {
                try_nb(self, node + width, &mut tail);
            }
            if x != 0 {
                try_nb(self, node - 1, &mut tail);
            }
            if x + 1 != width {
                try_nb(self, node + 1, &mut tail);
            }
        }
        None
    }
}

enum BfsVisit<R> {
    Found(R),
    Reject,
    Continue,
}

struct ConnectedComponents {
    ids: Vec<u16>,
    sizes: Vec<u32>,
}

impl ConnectedComponents {
    fn new(map: &GameMap) -> Self {
        let n = map.len();
        let width = map.width();
        let height = map.height();
        let last_row = (height.saturating_sub(1)) * width;
        let mut ids = vec![0u16; n];
        for i in 0..n {
            if !map.is_water(i as TileRef) {
                ids[i] = 0xffff;
            }
        }
        let mut sizes = vec![0u32];
        let mut queue = vec![0i32; n];
        let mut next_id: u16 = 0;
        for start in 0..n {
            if ids[start] != 0 {
                continue;
            }
            next_id = next_id.saturating_add(1);
            if next_id == 0xffff {
                break;
            }
            if sizes.len() <= next_id as usize {
                sizes.resize(next_id as usize + 1, 0);
            }
            let mut head = 0usize;
            let mut tail = 0usize;
            queue[tail] = start as i32;
            tail += 1;
            while head < tail {
                let seed = queue[head] as usize;
                head += 1;
                if ids[seed] != 0 {
                    continue;
                }
                let mut left = seed;
                let row_start = seed - (seed % width as usize);
                while left > row_start && ids[left - 1] == 0 {
                    left -= 1;
                }
                let mut right = seed;
                let row_end = row_start + width as usize - 1;
                while right < row_end && ids[right + 1] == 0 {
                    right += 1;
                }
                let span = (right - left + 1) as u32;
                sizes[next_id as usize] += span;
                for x in left..=right {
                    ids[x] = next_id;
                    if x as u32 >= width {
                        let above = x - width as usize;
                        if ids[above] == 0 {
                            queue[tail] = above as i32;
                            tail += 1;
                        }
                    }
                    if (x as u32) < last_row {
                        let below = x + width as usize;
                        if ids[below] == 0 {
                            queue[tail] = below as i32;
                            tail += 1;
                        }
                    }
                }
            }
        }
        Self { ids, sizes }
    }

    fn id(&self, tile: TileRef) -> u16 {
        self.ids.get(tile as usize).copied().unwrap_or(0)
    }
}

#[derive(Clone)]
struct AbstractNode {
    id: u32,
    x: u32,
    y: u32,
    tile: TileRef,
    component_id: u16,
}

#[derive(Clone)]
struct AbstractEdge {
    id: u32,
    node_a: u32,
    node_b: u32,
    cost: f32,
    cluster_x: u32,
    cluster_y: u32,
}

#[derive(Clone)]
struct Cluster {
    node_ids: Vec<u32>,
}

pub struct AbstractGraph {
    cluster_size: u32,
    clusters_x: u32,
    nodes: Vec<AbstractNode>,
    edges: Vec<AbstractEdge>,
    node_edge_ids: Vec<Vec<u32>>,
    clusters: Vec<Cluster>,
    components: ConnectedComponents,
}

impl AbstractGraph {
    fn node_count(&self) -> usize {
        self.nodes.len()
    }
    fn get_node(&self, id: u32) -> Option<&AbstractNode> {
        self.nodes.get(id as usize)
    }
    fn get_cluster(&self, cx: u32, cy: u32) -> Option<&Cluster> {
        self.clusters.get((cy * self.clusters_x + cx) as usize)
    }
    fn get_edge_between(&self, a: u32, b: u32) -> Option<&AbstractEdge> {
        let ids = self.node_edge_ids.get(a as usize)?;
        for &eid in ids {
            let e = &self.edges[eid as usize];
            if e.node_a == b || e.node_b == b {
                return Some(e);
            }
        }
        None
    }
    fn other(&self, e: &AbstractEdge, id: u32) -> u32 {
        if e.node_a == id {
            e.node_b
        } else {
            e.node_a
        }
    }
    fn component_id(&self, tile: TileRef) -> u16 {
        self.components.id(tile)
    }
}

struct GraphBuilder<'a> {
    map: &'a GameMap,
    graph: AbstractGraph,
    tile_to_node: std::collections::HashMap<TileRef, u32>,
    next_node: u32,
    cluster_size: u32,
    clusters_x: u32,
    clusters_y: u32,
}

impl<'a> GraphBuilder<'a> {
    fn new(map: &'a GameMap) -> Self {
        let cluster_size = CLUSTER_SIZE;
        let clusters_x = map.width().div_ceil(cluster_size);
        let clusters_y = map.height().div_ceil(cluster_size);
        Self {
            map,
            graph: AbstractGraph {
                cluster_size,
                clusters_x,
                nodes: Vec::new(),
                edges: Vec::new(),
                node_edge_ids: Vec::new(),
                clusters: (0..clusters_x * clusters_y)
                    .map(|_| Cluster { node_ids: Vec::new() })
                    .collect(),
                components: ConnectedComponents::new(map),
            },
            tile_to_node: std::collections::HashMap::new(),
            next_node: 0,
            cluster_size,
            clusters_x,
            clusters_y,
        }
    }

    fn get_or_create(&mut self, x: u32, y: u32) -> u32 {
        let tile = y * self.map.width() + x;
        if let Some(&id) = self.tile_to_node.get(&tile) {
            return id;
        }
        let id = self.next_node;
        self.next_node += 1;
        let component_id = self.graph.components.id(tile);
        self.graph.nodes.push(AbstractNode {
            id,
            x,
            y,
            tile,
            component_id,
        });
        self.graph.node_edge_ids.push(Vec::new());
        self.tile_to_node.insert(tile, id);
        id
    }

    fn add_to_cluster(&mut self, cx: u32, cy: u32, id: u32) {
        let key = (cy * self.clusters_x + cx) as usize;
        if !self.graph.clusters[key].node_ids.contains(&id) {
            self.graph.clusters[key].node_ids.push(id);
        }
    }

    fn find_vertical(&mut self, x: u32, base_y: u32) -> Vec<u32> {
        let mut nodes = Vec::new();
        let max_y = (base_y + self.cluster_size).min(self.map.height());
        let mut span_start: i32 = -1;
        for y in base_y..max_y {
            let tile = y * self.map.width() + x;
            let is_entrance = self.map.is_water(tile)
                && x + 1 < self.map.width()
                && self.map.is_water(y * self.map.width() + x + 1);
            if is_entrance {
                if span_start < 0 {
                    span_start = y as i32;
                }
            } else if span_start >= 0 {
                let mid_y = span_start + (y as i32 - span_start) / 2;
                span_start = -1;
                nodes.push(self.get_or_create(x, mid_y as u32));
            }
        }
        if span_start >= 0 {
            let mid_y = span_start + (max_y as i32 - span_start) / 2;
            nodes.push(self.get_or_create(x, mid_y as u32));
        }
        nodes
    }

    fn find_horizontal(&mut self, y: u32, base_x: u32) -> Vec<u32> {
        let mut nodes = Vec::new();
        let max_x = (base_x + self.cluster_size).min(self.map.width());
        let mut span_start: i32 = -1;
        for x in base_x..max_x {
            let tile = y * self.map.width() + x;
            let is_entrance = self.map.is_water(tile)
                && y + 1 < self.map.height()
                && self.map.is_water((y + 1) * self.map.width() + x);
            if is_entrance {
                if span_start < 0 {
                    span_start = x as i32;
                }
            } else if span_start >= 0 {
                let mid_x = span_start + (x as i32 - span_start) / 2;
                span_start = -1;
                nodes.push(self.get_or_create(mid_x as u32, y));
            }
        }
        if span_start >= 0 {
            let mid_x = span_start + (max_x as i32 - span_start) / 2;
            nodes.push(self.get_or_create(mid_x as u32, y));
        }
        nodes
    }

    fn process_nodes(&mut self) {
        for cy in 0..self.clusters_y {
            for cx in 0..self.clusters_x {
                let base_x = cx * self.cluster_size;
                let base_y = cy * self.cluster_size;
                if cx + 1 < self.clusters_x {
                    let edge_x = (base_x + self.cluster_size - 1).min(self.map.width() - 1);
                    let nodes = self.find_vertical(edge_x, base_y);
                    for id in nodes {
                        self.add_to_cluster(cx, cy, id);
                        self.add_to_cluster(cx + 1, cy, id);
                    }
                }
                if cy + 1 < self.clusters_y {
                    let edge_y = (base_y + self.cluster_size - 1).min(self.map.height() - 1);
                    let nodes = self.find_horizontal(edge_y, base_x);
                    for id in nodes {
                        self.add_to_cluster(cx, cy, id);
                        self.add_to_cluster(cx, cy + 1, id);
                    }
                }
            }
        }
    }
}

pub fn build_abstract_graph(map: &GameMap) -> AbstractGraph {
    let mut b = GraphBuilder::new(map);
    b.process_nodes();
    let cluster_size = b.cluster_size;
    let clusters_x = b.clusters_x;
    let clusters_y = b.clusters_y;
    let mut graph = b.graph;
    let width = map.width();
    let height = map.height();

    let mut edge_between: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::new();
    let mut bfs = BfsGrid::new(map.len());
    for cy in 0..clusters_y {
        for cx in 0..clusters_x {
            let Some(cluster) = graph.get_cluster(cx, cy) else {
                continue;
            };
            let node_ids = cluster.node_ids.clone();
            if node_ids.is_empty() {
                continue;
            }
            let cluster_min_x = cx * cluster_size;
            let cluster_min_y = cy * cluster_size;
            let cluster_max_x = (width - 1).min(cluster_min_x + cluster_size - 1);
            let cluster_max_y = (height - 1).min(cluster_min_y + cluster_size - 1);
            for i in 0..node_ids.len() {
                let from_id = node_ids[i];
                let from = graph.nodes[from_id as usize].clone();
                let mut targets = Vec::new();
                for j in (i + 1)..node_ids.len() {
                    let other = &graph.nodes[node_ids[j] as usize];
                    if from.component_id != other.component_id {
                        continue;
                    }
                    targets.push(other.clone());
                }
                if targets.is_empty() {
                    continue;
                }
                let mut tile_to_tid = std::collections::HashMap::new();
                let mut max_man = 0u32;
                for t in &targets {
                    tile_to_tid.insert(t.tile, t.id);
                    let dx = t.x.abs_diff(from.x);
                    let dy = t.y.abs_diff(from.y);
                    max_man = max_man.max(dx + dy);
                }
                let max_distance = max_man * 4;
                // Preserve BFS discovery order (official JS Map insertion) so
                // adjacency lists are process-stable and A* ties match.
                let mut reachable: Vec<(u32, u32)> = Vec::new();
                let found_needed = targets.len();
                let _ = bfs.search(
                    width,
                    height,
                    &[from.tile],
                    max_distance,
                    |t| map.is_water(t),
                    |tile, dist| {
                        let x = tile % width;
                        let y = tile / width;
                        let is_end = tile == from.tile || tile_to_tid.contains_key(&tile);
                        if !is_end
                            && (x < cluster_min_x
                                || x > cluster_max_x
                                || y < cluster_min_y
                                || y > cluster_max_y)
                        {
                            return BfsVisit::Reject;
                        }
                        if let Some(&tid) = tile_to_tid.get(&tile) {
                            if !reachable.iter().any(|(id, _)| *id == tid) {
                                reachable.push((tid, dist));
                            }
                            if reachable.len() == found_needed {
                                return BfsVisit::Found(dist);
                            }
                        }
                        BfsVisit::Continue
                    },
                );
                for (tid, cost) in reachable {
                    let (lo, hi) = if from_id < tid {
                        (from_id, tid)
                    } else {
                        (tid, from_id)
                    };
                    if let Some(&eid) = edge_between.get(&(lo, hi)) {
                        let e = &mut graph.edges[eid as usize];
                        if (cost as f32) < e.cost {
                            e.cost = cost as f32;
                            e.cluster_x = cx;
                            e.cluster_y = cy;
                        }
                        continue;
                    }
                    let eid = graph.edges.len() as u32;
                    graph.edges.push(AbstractEdge {
                        id: eid,
                        node_a: lo,
                        node_b: hi,
                        cost: cost as f32,
                        cluster_x: cx,
                        cluster_y: cy,
                    });
                    graph.node_edge_ids[lo as usize].push(eid);
                    graph.node_edge_ids[hi as usize].push(eid);
                    edge_between.insert((lo, hi), eid);
                }
            }
        }
    }
    graph
}

pub struct AStarWaterBounded {
    stamp: u32,
    closed_stamp: Vec<u32>,
    g_stamp: Vec<u32>,
    g_score: Vec<u32>,
    came_from: Vec<i32>,
    queue: MinHeap,
    width: u32,
    heuristic_weight: i32,
    max_iterations: i32,
    max_area: usize,
}

impl AStarWaterBounded {
    pub fn new(map: &GameMap, max_area: usize) -> Self {
        Self {
            stamp: 1,
            closed_stamp: vec![0; max_area],
            g_stamp: vec![0; max_area],
            g_score: vec![0; max_area],
            came_from: vec![-1; max_area],
            queue: MinHeap::with_capacity(max_area * 4),
            width: map.width(),
            heuristic_weight: 3,
            max_iterations: 100_000,
            max_area,
        }
    }

    pub fn search_bounded(
        &mut self,
        map: &GameMap,
        start: &[TileRef],
        goal: TileRef,
        min_x: u32,
        max_x: u32,
        min_y: u32,
        max_y: u32,
    ) -> Option<Vec<TileRef>> {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed_stamp.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let width = self.width;
        let bounds_w = max_x - min_x + 1;
        let bounds_h = max_y - min_y + 1;
        let num_local = (bounds_w * bounds_h) as usize;
        if num_local > self.max_area {
            return None;
        }
        let to_local = |tile: TileRef, clamp: bool| -> Option<u32> {
            let mut x = tile % width;
            let mut y = tile / width;
            if clamp {
                x = x.clamp(min_x, max_x);
                y = y.clamp(min_y, max_y);
            }
            Some((y - min_y) * bounds_w + (x - min_x))
        };
        let to_global = |local: u32| -> TileRef {
            let lx = local % bounds_w;
            let ly = local / bounds_w;
            (ly + min_y) * width + (lx + min_x)
        };
        let goal_local = to_local(goal, true)?;
        if goal_local as usize >= num_local {
            return None;
        }
        self.queue.clear();
        let s0 = *start.first()?;
        let start_x = (s0 % width) as i32;
        let start_y = (s0 / width) as i32;
        let goal_x = (goal % width) as i32;
        let goal_y = (goal / width) as i32;
        let dx_goal = goal_x - start_x;
        let dy_goal = goal_y - start_y;
        let cross_norm = 1.max(dx_goal.abs() + dy_goal.abs());
        let cross_tb = |nx: i32, ny: i32| -> i32 {
            let dx_n = nx - goal_x;
            let dy_n = ny - goal_y;
            let cross = (dx_goal * dy_n - dy_goal * dx_n).abs();
            ((cross * (COST_SCALE - 1)) / cross_norm) / cross_norm
        };
        let weight = self.heuristic_weight;
        let land_mask = 1u8 << LAND_BIT;
        for &s in start {
            let Some(sl) = to_local(s, true) else {
                continue;
            };
            if sl as usize >= num_local {
                continue;
            }
            self.g_score[sl as usize] = 0;
            self.g_stamp[sl as usize] = stamp;
            self.came_from[sl as usize] = -1;
            let sx = (s % width) as i32;
            let sy = (s / width) as i32;
            let h = weight * BASE_COST * ((sx - goal_x).abs() + (sy - goal_y).abs());
            self.queue.push(sl, h as f32);
        }
        let mut iterations = self.max_iterations;
        while !self.queue.is_empty() {
            iterations -= 1;
            if iterations <= 0 {
                return None;
            }
            let current_local = self.queue.pop();
            if self.closed_stamp[current_local as usize] == stamp {
                continue;
            }
            self.closed_stamp[current_local as usize] = stamp;
            if current_local == goal_local {
                return Some(self.build_local_path(goal_local, to_global, num_local));
            }
            let current_g = self.g_score[current_local as usize];
            let current = to_global(current_local);
            let current_x = (current % width) as i32;
            let current_y = (current / width) as i32;
            let try_nb = |this: &mut Self,
                          neighbor: TileRef,
                          neighbor_local: u32,
                          nx: i32,
                          ny: i32| {
                if neighbor_local as usize >= num_local {
                    return;
                }
                let terrain = map.terrain_byte(neighbor);
                if this.closed_stamp[neighbor_local as usize] == stamp {
                    return;
                }
                if neighbor != goal && (terrain & land_mask) != 0 {
                    return;
                }
                let mag = terrain & MAGNITUDE_MASK;
                let cost = BASE_COST + magnitude_penalty_bounded(mag);
                let tentative = current_g + cost as u32;
                if this.g_stamp[neighbor_local as usize] != stamp
                    || tentative < this.g_score[neighbor_local as usize]
                {
                    this.came_from[neighbor_local as usize] = current_local as i32;
                    this.g_score[neighbor_local as usize] = tentative;
                    this.g_stamp[neighbor_local as usize] = stamp;
                    let h = weight * BASE_COST * ((nx - goal_x).abs() + (ny - goal_y).abs());
                    let f = tentative as i32 + h + cross_tb(nx, ny);
                    this.queue.push(neighbor_local, f as f32);
                }
            };
            if current_y as u32 > min_y {
                try_nb(
                    self,
                    current - width,
                    current_local - bounds_w,
                    current_x,
                    current_y - 1,
                );
            }
            if (current_y as u32) < max_y {
                try_nb(
                    self,
                    current + width,
                    current_local + bounds_w,
                    current_x,
                    current_y + 1,
                );
            }
            if current_x as u32 > min_x {
                try_nb(self, current - 1, current_local - 1, current_x - 1, current_y);
            }
            if (current_x as u32) < max_x {
                try_nb(self, current + 1, current_local + 1, current_x + 1, current_y);
            }
        }
        None
    }

    fn build_local_path(
        &self,
        goal_local: u32,
        to_global: impl Fn(u32) -> TileRef,
        max_len: usize,
    ) -> Vec<TileRef> {
        let mut path = Vec::new();
        let mut current = goal_local as i32;
        let mut n = 0;
        while current != -1 && n < max_len {
            path.push(to_global(current as u32));
            current = self.came_from[current as usize];
            n += 1;
        }
        path.reverse();
        path
    }
}

struct AbstractAStar {
    stamp: u32,
    closed_stamp: Vec<u32>,
    g_stamp: Vec<u32>,
    g_score: Vec<f32>,
    came_from: Vec<i32>,
    queue: MinHeap,
    heuristic_weight: f32,
    max_iterations: i32,
}

impl AbstractAStar {
    fn new(graph: &AbstractGraph) -> Self {
        let n = graph.node_count();
        Self {
            stamp: 1,
            closed_stamp: vec![0; n],
            g_stamp: vec![0; n],
            g_score: vec![0.0; n],
            came_from: vec![-1; n],
            queue: MinHeap::with_capacity(n + graph.edges.len() * 2),
            heuristic_weight: 1.0,
            max_iterations: 100_000,
        }
    }

    fn find_path(&mut self, graph: &AbstractGraph, starts: &[u32], goal: u32) -> Option<Vec<u32>> {
        let goal_node = graph.get_node(goal)?;
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed_stamp.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        self.queue.clear();
        for &sid in starts {
            let Some(node) = graph.get_node(sid) else {
                continue;
            };
            self.g_score[sid as usize] = 0.0;
            self.g_stamp[sid as usize] = stamp;
            self.came_from[sid as usize] = -1;
            let h = self.heuristic_weight
                * ((node.x as i32 - goal_node.x as i32).abs()
                    + (node.y as i32 - goal_node.y as i32).abs()) as f32;
            self.queue.push(sid, h);
        }
        let mut iterations = self.max_iterations;
        while !self.queue.is_empty() {
            iterations -= 1;
            if iterations <= 0 {
                return None;
            }
            let current = self.queue.pop();
            if self.closed_stamp[current as usize] == stamp {
                continue;
            }
            self.closed_stamp[current as usize] = stamp;
            if current == goal {
                return self.build(goal);
            }
            let current_g = self.g_score[current as usize];
            let edges = &graph.node_edge_ids[current as usize];
            for &eid in edges {
                let edge = &graph.edges[eid as usize];
                let neighbor = graph.other(edge, current);
                if self.closed_stamp[neighbor as usize] == stamp {
                    continue;
                }
                let tentative = current_g + edge.cost;
                if self.g_stamp[neighbor as usize] != stamp
                    || tentative < self.g_score[neighbor as usize]
                {
                    self.came_from[neighbor as usize] = current as i32;
                    self.g_score[neighbor as usize] = tentative;
                    self.g_stamp[neighbor as usize] = stamp;
                    if let Some(nn) = graph.get_node(neighbor) {
                        let h = self.heuristic_weight
                            * ((nn.x as i32 - goal_node.x as i32).abs()
                                + (nn.y as i32 - goal_node.y as i32).abs())
                                as f32;
                        self.queue.push(neighbor, tentative + h);
                    }
                }
            }
        }
        None
    }

    fn build(&self, goal: u32) -> Option<Vec<u32>> {
        let mut path = Vec::new();
        let mut current = goal as i32;
        let max = self.came_from.len() as i32;
        while current != -1 {
            if current < 0 || current >= max {
                return None;
            }
            path.push(current as u32);
            if path.len() > self.came_from.len() {
                return None;
            }
            current = self.came_from[current as usize];
        }
        path.reverse();
        Some(path)
    }
}

pub struct WaterHpa {
    graph: AbstractGraph,
    abs_astar: AbstractAStar,
    local: AStarWaterBounded,
    local_multi: AStarWaterBounded,
    local_short: AStarWaterBounded,
    tile_bfs: BfsGrid,
}

impl WaterHpa {
    pub fn new(mini: &GameMap) -> Self {
        let graph = build_abstract_graph(mini);
        let cs = graph.cluster_size as usize;
        let abs_astar = AbstractAStar::new(&graph);
        Self {
            abs_astar,
            local: AStarWaterBounded::new(mini, cs * cs),
            local_multi: AStarWaterBounded::new(mini, (cs * 3) * (cs * 3)),
            local_short: AStarWaterBounded::new(mini, (SHORT_PATH_SIZE as usize).pow(2)),
            tile_bfs: BfsGrid::new(mini.len()),
            graph,
        }
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn component_id(&self, tile: TileRef) -> u16 {
        self.graph.component_id(tile)
    }

    fn find_nearest_node(&mut self, map: &GameMap, tile: TileRef) -> Option<u32> {
        let x = map.x(tile);
        let y = map.y(tile);
        let cs = self.graph.cluster_size;
        let cx = x / cs;
        let cy = y / cs;
        let cluster = self.graph.get_cluster(cx, cy)?;
        if cluster.node_ids.is_empty() {
            return None;
        }
        let candidates: Vec<u32> = cluster.node_ids.clone();
        let min_x = cx * cs;
        let min_y = cy * cs;
        let max_x = (map.width() - 1).min(min_x + cs - 1);
        let max_y = (map.height() - 1).min(min_y + cs - 1);
        let max_distance = cs * cs;
        let width = map.width();
        let height = map.height();
        self.tile_bfs.search(
            width,
            height,
            &[tile],
            max_distance,
            |t| map.is_water(t),
            |t, _dist| {
                let tx = map.x(t);
                let ty = map.y(t);
                for &id in &candidates {
                    let n = &self.graph.nodes[id as usize];
                    if n.x == tx && n.y == ty {
                        return BfsVisit::Found(id);
                    }
                }
                if tx < min_x || tx > max_x || ty < min_y || ty > max_y {
                    return BfsVisit::Reject;
                }
                BfsVisit::Continue
            },
        )
    }

    fn cluster_node(&self, map: &GameMap, tile: TileRef) -> Option<u32> {
        let x = map.x(tile);
        let y = map.y(tile);
        let cs = self.graph.cluster_size;
        let cluster = self.graph.get_cluster(x / cs, y / cs)?;
        if cluster.node_ids.is_empty() {
            return None;
        }
        let mut best = None;
        let mut best_d = u32::MAX;
        for &id in &cluster.node_ids {
            let n = &self.graph.nodes[id as usize];
            let d = n.x.abs_diff(x) + n.y.abs_diff(y);
            if d < best_d {
                best_d = d;
                best = Some(id);
            }
        }
        best
    }

    fn find_local(
        &mut self,
        map: &GameMap,
        from: TileRef,
        to: TileRef,
        cluster_x: u32,
        cluster_y: u32,
        multi: bool,
    ) -> Option<Vec<TileRef>> {
        let cs = self.graph.cluster_size;
        let (min_x, min_y, max_x, max_y) = if multi {
            (
                cluster_x.saturating_sub(1) * cs,
                cluster_y.saturating_sub(1) * cs,
                (map.width() - 1).min((cluster_x + 2) * cs - 1),
                (map.height() - 1).min((cluster_y + 2) * cs - 1),
            )
        } else {
            let min_x = cluster_x * cs;
            let min_y = cluster_y * cs;
            (
                min_x,
                min_y,
                (map.width() - 1).min(min_x + cs - 1),
                (map.height() - 1).min(min_y + cs - 1),
            )
        };
        let selected = if multi {
            &mut self.local_multi
        } else {
            &mut self.local
        };
        let mut path = selected.search_bounded(map, &[from], to, min_x, max_x, min_y, max_y)?;
        if path.is_empty() {
            return None;
        }
        if path[0] != from {
            path.insert(0, from);
        }
        if path.last().copied() != Some(to) {
            path.push(to);
        }
        Some(path)
    }

    fn find_single(&mut self, map: &GameMap, from: TileRef, to: TileRef) -> Option<Vec<TileRef>> {
        let dist = map.manhattan(from, to);
        let cs = self.graph.cluster_size;
        if dist <= cs {
            let sx = map.x(from) / cs;
            let sy = map.y(from) / cs;
            if let Some(p) = self.find_local(map, from, to, sx, sy, true) {
                return Some(p);
            }
        }
        let start = self.find_nearest_node(map, from)?;
        let end = self.find_nearest_node(map, to)?;
        if start == end {
            let n = self.graph.get_node(start)?;
            let cx = n.x / cs;
            let cy = n.y / cs;
            return self.find_local(map, from, to, cx, cy, true);
        }
        let node_path = self.abs_astar.find_path(&self.graph, &[start], end)?;
        let first = self.graph.get_node(node_path[0])?;
        let start_cx = map.x(from) / cs;
        let start_cy = map.y(from) / cs;
        let mut path = self.find_local(map, from, first.tile, start_cx, start_cy, false)?;
        for i in 0..node_path.len().saturating_sub(1) {
            let a = node_path[i];
            let b = node_path[i + 1];
            let edge = self.graph.get_edge_between(a, b)?;
            let from_tile = self.graph.get_node(a)?.tile;
            let to_tile = self.graph.get_node(b)?.tile;
            let seg = self.find_local(map, from_tile, to_tile, edge.cluster_x, edge.cluster_y, false)?;
            path.extend(seg.into_iter().skip(1));
        }
        let last = self.graph.get_node(*node_path.last()?)?;
        let end_cx = map.x(to) / cs;
        let end_cy = map.y(to) / cs;
        let end_seg = self.find_local(map, last.tile, to, end_cx, end_cy, false)?;
        path.extend(end_seg.into_iter().skip(1));
        Some(path)
    }

    fn try_short_multi(
        &mut self,
        map: &GameMap,
        sources: &[TileRef],
        target: TileRef,
    ) -> Option<Vec<TileRef>> {
        let candidates: Vec<TileRef> = sources
            .iter()
            .copied()
            .filter(|&s| map.manhattan(s, target) <= SHORT_PATH_THRESHOLD)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let mut min_x = map.x(target) as i32;
        let mut max_x = min_x;
        let mut min_y = map.y(target) as i32;
        let mut max_y = min_y;
        for &s in &candidates {
            let sx = map.x(s) as i32;
            let sy = map.y(s) as i32;
            min_x = min_x.min(sx);
            max_x = max_x.max(sx);
            min_y = min_y.min(sy);
            max_y = max_y.max(sy);
        }
        let min_x = 0.max(min_x - SHORT_PATH_PADDING) as u32;
        let max_x = ((map.width() as i32 - 1).min(max_x + SHORT_PATH_PADDING)) as u32;
        let min_y = 0.max(min_y - SHORT_PATH_PADDING) as u32;
        let max_y = ((map.height() as i32 - 1).min(max_y + SHORT_PATH_PADDING)) as u32;
        self.local_short
            .search_bounded(map, &candidates, target, min_x, max_x, min_y, max_y)
    }

    pub fn find_path(&mut self, map: &GameMap, from: &[TileRef], to: TileRef) -> Option<Vec<TileRef>> {
        if from.is_empty() {
            return None;
        }
        if from.len() == 1 {
            return self.find_single(map, from[0], to);
        }
        if let Some(p) = self.try_short_multi(map, from, to) {
            return Some(p);
        }
        let target = self.cluster_node(map, to)?;
        let mut node_to_source: std::collections::HashMap<u32, TileRef> =
            std::collections::HashMap::new();
        let mut node_to_dist: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
        for &s in from {
            let Some(node) = self.cluster_node(map, s) else {
                continue;
            };
            let n = self.graph.get_node(node)?;
            let d = n.x.abs_diff(map.x(s)) + n.y.abs_diff(map.y(s));
            let prev = node_to_dist.get(&node).copied();
            if prev.is_none() || d < prev.unwrap() {
                node_to_source.insert(node, s);
                node_to_dist.insert(node, d);
            }
        }
        if node_to_source.is_empty() {
            return None;
        }
        let mut ids: Vec<u32> = Vec::new();
        for &s in from {
            if let Some(node) = self.cluster_node(map, s) {
                if node_to_source.contains_key(&node) && !ids.contains(&node) {
                    ids.push(node);
                }
            }
        }
        let node_path = self.abs_astar.find_path(&self.graph, &ids, target)?;
        let winner = *node_to_source.get(&node_path[0])?;
        self.find_single(map, winner, to)
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

fn shore_coerce(
    map: &GameMap,
    from: &[TileRef],
    to: TileRef,
) -> Option<(Vec<TileRef>, TileRef, std::collections::HashMap<TileRef, TileRef>, Option<TileRef>)> {
    let mut water_from = Vec::new();
    let mut water_to_original = std::collections::HashMap::new();
    for &f in from {
        if map.is_water(f) {
            water_from.push(f);
            water_to_original.remove(&f);
        } else if let Some(w) = best_water_neighbor(map, f) {
            water_from.push(w);
            water_to_original.insert(w, f);
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
    Some((water_from, water_to, water_to_original, original_to))
}

fn restore_shores(
    mut path: Vec<TileRef>,
    water_to_original: &std::collections::HashMap<TileRef, TileRef>,
    original_to: Option<TileRef>,
) -> Vec<TileRef> {
    if let Some(&shore) = path.first().and_then(|p| water_to_original.get(p)) {
        path.insert(0, shore);
    }
    if let Some(orig) = original_to {
        if path.last().copied() != Some(orig) {
            path.push(orig);
        }
    }
    path
}

fn manhattan(width: u32, a: TileRef, b: TileRef) -> u32 {
    let ax = a % width;
    let ay = a / width;
    let bx = b % width;
    let by = b / width;
    ax.abs_diff(bx) + ay.abs_diff(by)
}

fn los_can_see(
    map: &GameMap,
    from: TileRef,
    to: TileRef,
    min_mag: u8,
) -> bool {
    let width = map.width();
    let x0 = (from % width) as i32;
    let y0 = (from / width) as i32;
    let x1 = (to % width) as i32;
    let y1 = (to / width) as i32;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x0;
    let mut y = y0;
    for _ in 0..100_000 {
        let tile = (y as u32) * width + x as u32;
        if !map.is_water(tile) {
            return false;
        }
        if map.magnitude(tile) < min_mag {
            return false;
        }
        if x == x1 && y == y1 {
            return true;
        }
        let e2 = 2 * err;
        let move_x = e2 > -dy;
        let move_y = e2 < dx;
        if move_x && move_y {
            x += sx;
            err -= dy;
            let mid = (y as u32) * width + x as u32;
            if !map.is_water(mid) || map.magnitude(mid) < min_mag {
                x -= sx;
                err += dy;
                y += sy;
                err += dx;
                let alt = (y as u32) * width + x as u32;
                if !map.is_water(alt) || map.magnitude(alt) < min_mag {
                    return false;
                }
                x += sx;
                err -= dy;
            } else {
                y += sy;
                err += dx;
            }
        } else {
            if move_x {
                x += sx;
                err -= dy;
            }
            if move_y {
                y += sy;
                err += dx;
            }
        }
    }
    false
}

fn los_trace(map: &GameMap, from: TileRef, to: TileRef) -> Option<Vec<TileRef>> {
    let width = map.width();
    let x0 = (from % width) as i32;
    let y0 = (from / width) as i32;
    let x1 = (to % width) as i32;
    let y1 = (to / width) as i32;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x0;
    let mut y = y0;
    let mut tiles = Vec::new();
    for _ in 0..100_000 {
        let tile = (y as u32) * width + x as u32;
        if !map.is_water(tile) {
            return None;
        }
        tiles.push(tile);
        if x == x1 && y == y1 {
            return Some(tiles);
        }
        let e2 = 2 * err;
        let move_x = e2 > -dy;
        let move_y = e2 < dx;
        if move_x && move_y {
            x += sx;
            err -= dy;
            let mid = (y as u32) * width + x as u32;
            if !map.is_water(mid) {
                x -= sx;
                err += dy;
                y += sy;
                err += dx;
                let alt = (y as u32) * width + x as u32;
                if !map.is_water(alt) {
                    return None;
                }
                tiles.push(alt);
                x += sx;
                err -= dy;
            } else {
                tiles.push(mid);
                y += sy;
                err += dx;
            }
        } else {
            if move_x {
                x += sx;
                err -= dy;
            }
            if move_y {
                y += sy;
                err += dx;
            }
        }
    }
    None
}

fn los_smooth(map: &GameMap, path: &[TileRef], min_mag: u8) -> Vec<TileRef> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut result = vec![path[0]];
    let mut current = 0;
    while current < path.len() - 1 {
        let mut lo = current + 1;
        let mut hi = path.len() - 1;
        let mut farthest = lo;
        while lo <= hi {
            let mid = (lo + hi) >> 1;
            if los_can_see(map, path[current], path[mid], min_mag) {
                farthest = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if farthest > current + 1 {
            if let Some(trace) = los_trace(map, path[current], path[farthest]) {
                for t in trace.iter().skip(1).take(trace.len().saturating_sub(2)) {
                    result.push(*t);
                }
            }
        }
        current = farthest;
        if current < path.len() - 1 {
            result.push(path[current]);
        }
    }
    result.push(*path.last().unwrap());
    result
}

fn find_tile_at_distance(map: &GameMap, path: &[TileRef], start: usize, dist: u32, forward: bool) -> usize {
    let mut cum = 0u32;
    let mut idx = start;
    if forward {
        while idx + 1 < path.len() && cum < dist {
            cum += map.manhattan(path[idx], path[idx + 1]);
            idx += 1;
        }
    } else {
        while idx > 0 && cum < dist {
            cum += map.manhattan(path[idx], path[idx - 1]);
            idx -= 1;
        }
    }
    idx
}

fn refine_segment(
    bounded: &mut AStarWaterBounded,
    map: &GameMap,
    from: TileRef,
    to: TileRef,
) -> Option<Vec<TileRef>> {
    let x0 = map.x(from) as i32;
    let y0 = map.y(from) as i32;
    let x1 = map.x(to) as i32;
    let y1 = map.y(to) as i32;
    let pad = 10;
    let min_x = 0.max(x0.min(x1) - pad) as u32;
    let max_x = ((map.width() as i32 - 1).min(x0.max(x1) + pad)) as u32;
    let min_y = 0.max(y0.min(y1) - pad) as u32;
    let max_y = ((map.height() as i32 - 1).min(y0.max(y1) + pad)) as u32;
    bounded.search_bounded(map, &[from], to, min_x, max_x, min_y, max_y)
}

fn refine_endpoints(
    bounded: &mut AStarWaterBounded,
    map: &GameMap,
    path: &[TileRef],
) -> Vec<TileRef> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut result = path.to_vec();
    let start_end = find_tile_at_distance(map, path, 0, ENDPOINT_REFINEMENT_TILES, true);
    if start_end > 1 {
        if let Some(seg) = refine_segment(bounded, map, path[0], path[start_end]) {
            if !seg.is_empty() {
                result = seg[..seg.len() - 1]
                    .iter()
                    .copied()
                    .chain(result[start_end..].iter().copied())
                    .collect();
            }
        }
    }
    let end_start = find_tile_at_distance(
        map,
        &result,
        result.len() - 1,
        ENDPOINT_REFINEMENT_TILES,
        false,
    );
    if end_start + 2 < result.len() {
        if let Some(mut seg) = refine_segment(bounded, map, result[result.len() - 1], result[end_start]) {
            if !seg.is_empty() {
                seg.reverse();
                result = result[..end_start]
                    .iter()
                    .copied()
                    .chain(seg.into_iter())
                    .collect();
            }
        }
    }
    result
}

fn smooth_path(bounded: &mut AStarWaterBounded, map: &GameMap, path: Vec<TileRef>) -> Vec<TileRef> {
    if path.len() <= 2 {
        return path;
    }
    let s1 = los_smooth(map, &path, LOS_MIN_MAGNITUDE_PASS1);
    let s2 = refine_endpoints(bounded, map, &s1);
    los_smooth(map, &s2, LOS_MIN_MAGNITUDE_PASS2)
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

/// Official WaterPathFinder chain (simple mini A* or HPA).
pub struct WaterChain {
    simple: crate::path::AStarWater,
    hpa: Option<WaterHpa>,
    smooth: AStarWaterBounded,
}

impl WaterChain {
    pub fn new(mini: &GameMap) -> Self {
        let hpa = WaterHpa::new(mini);
        let use_hpa = hpa.node_count() >= HPA_NODE_THRESHOLD;
        Self {
            simple: crate::path::AStarWater::new(mini),
            hpa: if use_hpa { Some(hpa) } else { None },
            smooth: AStarWaterBounded::new(mini, LOCAL_ASTAR_MAX_AREA),
        }
    }

    pub fn uses_hpa(&self) -> bool {
        self.hpa.is_some()
    }

    pub fn find_path(
        &mut self,
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
        let coerced = shore_coerce(mini, &mini_from, mini_to)?;
        let (water_from, water_to, orig_map, orig_to) = coerced;
        let mini_path = if let Some(hpa) = &mut self.hpa {
            let to_c = hpa.component_id(water_to);
            let valid: Vec<TileRef> = water_from
                .into_iter()
                .filter(|&f| hpa.component_id(f) == to_c)
                .collect();
            if valid.is_empty() {
                return None;
            }
            let raw = hpa.find_path(mini, &valid, water_to)?;
            smooth_path(&mut self.smooth, mini, raw)
        } else {
            self.simple.find_path(mini, &water_from, water_to)?
        };
        let mini_path = restore_shores(mini_path, &orig_map, orig_to);
        if mini_path.is_empty() {
            return None;
        }
        let mut upscaled = upscale_path(mini, map, &mini_path);
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
}

/// Official refineStartTile (bounded A* on the main map).
pub fn refine_start_tile_bounded(
    bounded: &mut AStarWaterBounded,
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
    let min_x = 0.max(cand_min_x.min(wpx) - PADDING) as u32;
    let max_x = ((map.width() as i32 - 1).min(cand_max_x.max(wpx) + PADDING)) as u32;
    let min_y = 0.max(cand_min_y.min(wpy) - PADDING) as u32;
    let max_y = ((map.height() as i32 - 1).min(cand_max_y.max(wpy) + PADDING)) as u32;
    let area = (max_x as i32 - min_x as i32 + 1) * (max_y as i32 - min_y as i32 + 1);
    if area > REFINE_MAX {
        return best_tile;
    }
    bounded
        .search_bounded(map, &candidates, waypoint, min_x, max_x, min_y, max_y)
        .and_then(|p| p.first().copied())
        .unwrap_or(best_tile)
}

pub fn new_refine_astar(map: &GameMap) -> AStarWaterBounded {
    AStarWaterBounded::new(map, LOCAL_ASTAR_MAX_AREA)
}

#[allow(dead_code)]
fn _unused_helpers() {
    let _ = manhattan;
    let _ = magnitude_penalty_simple;
}
