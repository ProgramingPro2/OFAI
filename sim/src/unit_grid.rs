//! Official UnitGrid: 100-tile cells, insertion-ordered sets per type.

use std::collections::HashMap;

use indexmap::IndexSet;

use crate::map::{GameMap, TileRef};
use crate::units::{Unit, UnitType};

const CELL: u32 = 100;

pub struct UnitGrid {
    cells_x: usize,
    cells_y: usize,
    grid: Vec<HashMap<UnitType, IndexSet<u32>>>,
}

impl UnitGrid {
    pub fn new(map: &GameMap) -> Self {
        let cells_x = ((map.width() + CELL - 1) / CELL) as usize;
        let cells_y = ((map.height() + CELL - 1) / CELL) as usize;
        Self {
            cells_x,
            cells_y,
            grid: vec![HashMap::new(); cells_x.max(1) * cells_y.max(1)],
        }
    }

    fn coords(map: &GameMap, tile: TileRef) -> (i32, i32) {
        (
            (map.x(tile) / CELL) as i32,
            (map.y(tile) / CELL) as i32,
        )
    }

    fn idx(&self, gx: i32, gy: i32) -> Option<usize> {
        if gx < 0 || gy < 0 {
            return None;
        }
        let gx = gx as usize;
        let gy = gy as usize;
        if gx >= self.cells_x || gy >= self.cells_y {
            return None;
        }
        Some(gy * self.cells_x + gx)
    }

    pub fn add(&mut self, map: &GameMap, unit: &Unit) {
        let (gx, gy) = Self::coords(map, unit.tile);
        if let Some(i) = self.idx(gx, gy) {
            self.grid[i].entry(unit.ty).or_default().insert(unit.id);
        }
    }

    pub fn remove(&mut self, map: &GameMap, unit: &Unit) {
        self.remove_at(map, unit.id, unit.ty, unit.last_tile);
        self.remove_at(map, unit.id, unit.ty, unit.tile);
    }

    fn remove_at(&mut self, map: &GameMap, id: u32, ty: UnitType, tile: TileRef) {
        let (gx, gy) = Self::coords(map, tile);
        if let Some(i) = self.idx(gx, gy) {
            if let Some(set) = self.grid[i].get_mut(&ty) {
                set.shift_remove(&id);
            }
        }
    }

    pub fn update_cell(&mut self, map: &GameMap, unit: &Unit) {
        let (ogx, ogy) = Self::coords(map, unit.last_tile);
        let (ngx, ngy) = Self::coords(map, unit.tile);
        if ogx != ngx || ogy != ngy {
            self.remove_at(map, unit.id, unit.ty, unit.last_tile);
            if let Some(i) = self.idx(ngx, ngy) {
                self.grid[i].entry(unit.ty).or_default().insert(unit.id);
            }
        }
    }

    fn cells_in_range(
        &self,
        map: &GameMap,
        tile: TileRef,
        range: u32,
    ) -> (i32, i32, i32, i32) {
        let x = map.x(tile) as f64;
        let y = map.y(tile) as f64;
        let cell = CELL as f64;
        let (gx, gy) = Self::coords(map, tile);
        let start_gx = (gx as f64 - ((range as f64 - (x % cell)) / cell).ceil()).max(0.0) as i32;
        let end_gx = (gx as f64 + ((range as f64 - (cell - (x % cell))) / cell).ceil())
            .min(self.cells_x as f64 - 1.0) as i32;
        let start_gy = (gy as f64 - ((range as f64 - (y % cell)) / cell).ceil()).max(0.0) as i32;
        let end_gy = (gy as f64 + ((range as f64 - (cell - (y % cell))) / cell).ceil())
            .min(self.cells_y as f64 - 1.0) as i32;
        (start_gx, end_gx, start_gy, end_gy)
    }

    pub fn nearby(
        &self,
        map: &GameMap,
        tile: TileRef,
        range: u32,
        types: &[UnitType],
        include_construction: bool,
        units: &[Unit],
    ) -> Vec<(u32, i64)> {
        let (sx, ex, sy, ey) = self.cells_in_range(map, tile, range);
        let r2 = range as i64 * range as i64;
        let cx = map.x(tile) as i64;
        let cy = map.y(tile) as i64;
        let mut out = Vec::new();
        for gy in sy..=ey {
            for gx in sx..=ex {
                let Some(i) = self.idx(gx, gy) else { continue };
                for ty in types {
                    let Some(set) = self.grid[i].get(ty) else { continue };
                    for id in set {
                        let Some(u) = units.iter().find(|u| u.id == *id) else {
                            continue;
                        };
                        if !u.active {
                            continue;
                        }
                        if !include_construction && u.under_construction {
                            continue;
                        }
                        let dx = map.x(u.tile) as i64 - cx;
                        let dy = map.y(u.tile) as i64 - cy;
                        let d2 = dx * dx + dy * dy;
                        if d2 <= r2 {
                            out.push((*id, d2));
                        }
                    }
                }
            }
        }
        out
    }

    pub fn has_nearby(
        &self,
        map: &GameMap,
        tile: TileRef,
        range: u32,
        ty: UnitType,
        include_construction: bool,
        units: &[Unit],
    ) -> bool {
        !self
            .nearby(map, tile, range, &[ty], include_construction, units)
            .is_empty()
    }
}
