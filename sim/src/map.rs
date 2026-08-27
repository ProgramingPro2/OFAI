//! Packed OpenFront GameMap (terrain byte + uint16 owner/fallout/defense).

use std::sync::Arc;

use crate::spec::TerrainKind;

pub type TileRef = u32;

const IS_LAND_BIT: u8 = 7;
const SHORELINE_BIT: u8 = 6;
const OCEAN_BIT: u8 = 5;
const MAGNITUDE_MASK: u8 = 0x1f;
const IMPASSABLE_MAGNITUDE: u8 = 31;
const PLAYER_ID_MASK: u16 = 0xfff;
const FALLOUT_BIT: u16 = 13;
#[allow(dead_code)]
const DEFENSE_BONUS_BIT: u16 = 14;

#[derive(Clone)]
pub struct GameMap {
    width: u32,
    height: u32,
    num_land_tiles: u32,
    terrain: Arc<[u8]>,
    state: Vec<u16>,
    num_fallout: u32,
}

impl GameMap {
    pub fn new(width: u32, height: u32, terrain: Vec<u8>, num_land_tiles: u32) -> Result<Self, String> {
        Self::from_shared(width, height, Arc::<[u8]>::from(terrain), num_land_tiles)
    }

    pub fn from_shared(
        width: u32,
        height: u32,
        terrain: Arc<[u8]>,
        num_land_tiles: u32,
    ) -> Result<Self, String> {
        let n = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "map dimensions overflow".to_string())?;
        if terrain.len() != n {
            return Err(format!(
                "terrain length {} != {}x{}",
                terrain.len(),
                width,
                height
            ));
        }
        Ok(Self {
            width,
            height,
            num_land_tiles,
            terrain,
            state: vec![0; n],
            num_fallout: 0,
        })
    }

    pub fn terrain_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.terrain)
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn num_land_tiles(&self) -> u32 {
        self.num_land_tiles
    }
    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn ref_xy(&self, x: u32, y: u32) -> Result<TileRef, String> {
        if x >= self.width || y >= self.height {
            return Err(format!("invalid coordinates {x},{y}"));
        }
        Ok(y * self.width + x)
    }

    pub fn is_valid_ref(&self, r: i32) -> bool {
        r >= 0 && (r as usize) < self.state.len()
    }

    pub fn x(&self, r: TileRef) -> u32 {
        r % self.width
    }
    pub fn y(&self, r: TileRef) -> u32 {
        r / self.width
    }

    pub fn is_land(&self, r: TileRef) -> bool {
        self.terrain[r as usize] & (1 << IS_LAND_BIT) != 0
    }
    pub fn is_impassable(&self, r: TileRef) -> bool {
        self.is_land(r)
            && (self.terrain[r as usize] & MAGNITUDE_MASK) == IMPASSABLE_MAGNITUDE
    }
    pub fn is_ocean(&self, r: TileRef) -> bool {
        self.terrain[r as usize] & (1 << OCEAN_BIT) != 0
    }
    pub fn is_shoreline(&self, r: TileRef) -> bool {
        self.terrain[r as usize] & (1 << SHORELINE_BIT) != 0
    }
    pub fn is_water(&self, r: TileRef) -> bool {
        !self.is_land(r)
    }
    pub fn is_shore(&self, r: TileRef) -> bool {
        self.is_land(r) && self.is_shoreline(r)
    }
    pub fn magnitude(&self, r: TileRef) -> u8 {
        self.terrain[r as usize] & MAGNITUDE_MASK
    }
    pub fn terrain_byte(&self, r: TileRef) -> u8 {
        self.terrain[r as usize]
    }

    pub fn terrain_type(&self, r: TileRef) -> TerrainKind {
        if self.is_land(r) {
            let mag = self.magnitude(r);
            if mag >= IMPASSABLE_MAGNITUDE {
                TerrainKind::Impassable
            } else if mag < 10 {
                TerrainKind::Plains
            } else if mag < 20 {
                TerrainKind::Highland
            } else {
                TerrainKind::Mountain
            }
        } else {
            TerrainKind::Ocean
        }
    }

    pub fn owner_id(&self, r: TileRef) -> u16 {
        self.state[r as usize] & PLAYER_ID_MASK
    }
    pub fn has_owner(&self, r: TileRef) -> bool {
        self.owner_id(r) != 0
    }
    pub fn set_owner_id(&mut self, r: TileRef, id: u16) {
        let s = &mut self.state[r as usize];
        *s = (*s & !PLAYER_ID_MASK) | (id & PLAYER_ID_MASK);
    }

    pub fn has_fallout(&self, r: TileRef) -> bool {
        self.state[r as usize] & (1 << FALLOUT_BIT) != 0
    }
    pub fn set_fallout(&mut self, r: TileRef, value: bool) {
        let existing = self.has_fallout(r);
        if value && !existing {
            self.num_fallout += 1;
            self.state[r as usize] |= 1 << FALLOUT_BIT;
        } else if !value && existing {
            self.num_fallout -= 1;
            self.state[r as usize] &= !(1 << FALLOUT_BIT);
        }
    }
    pub fn num_tiles_with_fallout(&self) -> u32 {
        self.num_fallout
    }

    pub fn tile_state(&self, r: TileRef) -> u16 {
        self.state[r as usize]
    }
    pub fn tile_state_buffer(&self) -> &[u16] {
        &self.state
    }

    pub fn is_on_edge(&self, r: TileRef) -> bool {
        let x = self.x(r);
        let y = self.y(r);
        x == 0 || x + 1 == self.width || y == 0 || y + 1 == self.height
    }

    pub fn is_border(&self, r: TileRef) -> bool {
        let owner = self.owner_id(r);
        let mut nbuf = [0u32; 4];
        let n = self.neighbors4(r, &mut nbuf);
        for i in 0..n {
            if self.owner_id(nbuf[i]) != owner {
                return true;
            }
        }
        false
    }

    pub fn is_ocean_shore(&self, r: TileRef) -> bool {
        if !self.is_land(r) {
            return false;
        }
        let mut nbuf = [0u32; 4];
        let n = self.neighbors4(r, &mut nbuf);
        for i in 0..n {
            if self.is_ocean(nbuf[i]) {
                return true;
            }
        }
        false
    }

    /// Cardinal neighbors in official N, S, W, E order.
    pub fn neighbors4(&self, r: TileRef, out: &mut [TileRef; 4]) -> usize {
        let w = self.width;
        let x = r % w;
        let mut n = 0;
        if r >= w {
            out[n] = r - w;
            n += 1;
        }
        if r < (self.height - 1) * w {
            out[n] = r + w;
            n += 1;
        }
        if x != 0 {
            out[n] = r - 1;
            n += 1;
        }
        if x + 1 != w {
            out[n] = r + 1;
            n += 1;
        }
        n
    }

    pub fn for_each_neighbor_diag(&self, r: TileRef, mut cb: impl FnMut(TileRef)) {
        let w = self.width;
        let x = r % w;
        let has_n = r >= w;
        let has_s = r < (self.height - 1) * w;
        if x != 0 {
            if has_n {
                cb(r - 1 - w);
            }
            cb(r - 1);
            if has_s {
                cb(r - 1 + w);
            }
        }
        if has_n {
            cb(r - w);
        }
        if has_s {
            cb(r + w);
        }
        if x + 1 != w {
            if has_n {
                cb(r + 1 - w);
            }
            cb(r + 1);
            if has_s {
                cb(r + 1 + w);
            }
        }
    }

    pub fn manhattan(&self, a: TileRef, b: TileRef) -> u32 {
        self.x(a).abs_diff(self.x(b)) + self.y(a).abs_diff(self.y(b))
    }

    /// Official BFS/DFS (stack, N/S/W/E), insertion order preserved.
    pub fn bfs_euclid_center(&self, root: TileRef, dist: f64) -> Vec<TileRef> {
        let dist2 = dist * dist;
        let root_x = self.x(root) as f64 - 0.5;
        let root_y = self.y(root) as f64 - 0.5;
        let filter = |n: TileRef| {
            let dx = self.x(n) as f64 - root_x;
            let dy = self.y(n) as f64 - root_y;
            dx * dx + dy * dy <= dist2
        };
        let mut seen = vec![false; self.state.len()];
        let mut order = Vec::new();
        let mut stack = Vec::new();
        if filter(root) {
            seen[root as usize] = true;
            order.push(root);
            stack.push(root);
        }
        let mut nbuf = [0u32; 4];
        while let Some(curr) = stack.pop() {
            let n = self.neighbors4(curr, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if !seen[nb as usize] && filter(nb) {
                    seen[nb as usize] = true;
                    order.push(nb);
                    stack.push(nb);
                }
            }
        }
        order
    }

    pub fn bfs_manhattan(&self, root: TileRef, dist: u32) -> Vec<TileRef> {
        let mut seen = vec![false; self.state.len()];
        let mut order = Vec::new();
        let mut stack = Vec::new();
        seen[root as usize] = true;
        order.push(root);
        stack.push(root);
        let mut nbuf = [0u32; 4];
        while let Some(curr) = stack.pop() {
            let n = self.neighbors4(curr, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if !seen[nb as usize] && self.manhattan(root, nb) <= dist {
                    seen[nb as usize] = true;
                    order.push(nb);
                    stack.push(nb);
                }
            }
        }
        order
    }
}

/// Connected water-body labels (cardinal). IDs are local; only equality matters.
#[derive(Clone)]
pub struct WaterComponents {
    ids: Arc<[i32]>,
    sizes: Arc<[u32]>,
}

impl WaterComponents {
    pub fn compute(map: &GameMap) -> Self {
        let n = map.len();
        let mut ids = vec![-1i32; n];
        let mut sizes = vec![0u32];
        let mut next = 1i32;
        let mut nbuf = [0u32; 4];
        let mut stack = Vec::new();
        for r in 0..n as u32 {
            if !map.is_water(r) || ids[r as usize] != -1 {
                continue;
            }
            let id = next;
            next += 1;
            sizes.push(0);
            ids[r as usize] = id;
            sizes[id as usize] += 1;
            stack.push(r);
            while let Some(cur) = stack.pop() {
                let cnt = map.neighbors4(cur, &mut nbuf);
                for i in 0..cnt {
                    let nb = nbuf[i];
                    if map.is_water(nb) && ids[nb as usize] == -1 {
                        ids[nb as usize] = id;
                        sizes[id as usize] += 1;
                        stack.push(nb);
                    }
                }
            }
        }
        Self {
            ids: Arc::from(ids),
            sizes: Arc::from(sizes),
        }
    }

    pub fn water_id(&self, tile: TileRef) -> Option<i32> {
        let id = self.ids.get(tile as usize).copied().unwrap_or(-1);
        if id > 0 {
            Some(id)
        } else {
            None
        }
    }

    pub fn component_size(&self, id: i32) -> u32 {
        self.sizes.get(id as usize).copied().unwrap_or(0)
    }

    pub fn of_tile(&self, map: &GameMap, tile: TileRef) -> Option<i32> {
        if map.is_water(tile) {
            let id = self.ids[tile as usize];
            return if id > 0 { Some(id) } else { None };
        }
        let mut nbuf = [0u32; 4];
        let n = map.neighbors4(tile, &mut nbuf);
        for i in 0..n {
            if map.is_water(nbuf[i]) {
                let id = self.ids[nbuf[i] as usize];
                if id > 0 {
                    return Some(id);
                }
            }
        }
        for i in 0..n {
            let mut nbuf2 = [0u32; 4];
            let n2 = map.neighbors4(nbuf[i], &mut nbuf2);
            for j in 0..n2 {
                if map.is_water(nbuf2[j]) {
                    let id = self.ids[nbuf2[j] as usize];
                    if id > 0 {
                        return Some(id);
                    }
                }
            }
        }
        None
    }
}
