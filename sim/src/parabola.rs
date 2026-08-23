//! Official DistanceBasedBezierCurve + ParabolaUniversalPathFinder.

use crate::map::{GameMap, TileRef};

const PARABOLA_MIN_HEIGHT: f64 = 50.0;

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

pub struct ParabolaPath {
    points: Vec<Point>,
    current: usize,
    traveled: f64,
}

impl ParabolaPath {
    pub fn new(map: &GameMap, from: TileRef, to: TileRef, increment: u32, direction_up: bool) -> Self {
        let p0 = Point {
            x: map.x(from) as f64,
            y: map.y(from) as f64,
        };
        let p3 = Point {
            x: map.x(to) as f64,
            y: map.y(to) as f64,
        };
        let dx = p3.x - p0.x;
        let dy = p3.y - p0.y;
        let distance = (dx * dx + dy * dy).sqrt();
        let max_h = distance / 3.0;
        let max_h = if max_h < PARABOLA_MIN_HEIGHT {
            PARABOLA_MIN_HEIGHT
        } else {
            max_h
        };
        let hm = if direction_up { -1.0 } else { 1.0 };
        let mh = map.height() as f64;
        let p1 = Point {
            x: p0.x + dx / 4.0,
            y: within(p0.y + dy / 4.0 + hm * max_h, 0.0, mh - 1.0),
        };
        let p2 = Point {
            x: p0.x + (dx * 3.0) / 4.0,
            y: within(p0.y + (dy * 3.0) / 4.0 + hm * max_h, 0.0, mh - 1.0),
        };
        Self {
            points: compute_points(p0, p1, p2, p3, increment as f64, 0.002),
            current: 0,
            traveled: 0.0,
        }
    }

    pub fn increment(&mut self, distance: f64) -> Option<(f64, f64)> {
        self.traveled += distance;
        while self.current < self.points.len().saturating_sub(1)
            && dist_up_to(&self.points, self.current + 1) < self.traveled
        {
            self.current += 1;
        }
        if self.current >= self.points.len().saturating_sub(1) {
            return None;
        }
        let p = self.points[self.current];
        Some((p.x, p.y))
    }

    pub fn current_index(&self) -> usize {
        self.current
    }

    pub fn all_tiles(&self, map: &GameMap) -> Vec<TileRef> {
        self.points
            .iter()
            .filter_map(|p| map.ref_xy(p.x.floor() as u32, p.y.floor() as u32).ok())
            .collect()
    }
}

fn bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: f64) -> Point {
    let t1 = 1.0 - t;
    let tt = t * t;
    let ttt = tt * t;
    let tt1 = t1 * t1;
    let ttt1 = tt1 * t1;
    Point {
        x: ttt1 * p0.x + 3.0 * tt1 * t * p1.x + 3.0 * t1 * tt * p2.x + ttt * p3.x,
        y: ttt1 * p0.y + 3.0 * tt1 * t * p1.y + 3.0 * t1 * tt * p2.y + ttt * p3.y,
    }
}

fn compute_points(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    spacing: f64,
    precision: f64,
) -> Vec<Point> {
    let mut pts = vec![p0];
    let mut t = 0.0;
    let mut prev = p0;
    let mut cum = 0.0;
    while t < 1.0 {
        t = (t + precision).min(1.0);
        let cur = bezier(p0, p1, p2, p3, t);
        let dx = cur.x - prev.x;
        let dy = cur.y - prev.y;
        cum += (dx * dx + dy * dy).sqrt();
        if cum >= spacing {
            pts.push(cur);
            cum -= spacing;
        }
        prev = cur;
    }
    let last = bezier(p0, p1, p2, p3, 1.0);
    if pts.last().map(|p| p.x != last.x || p.y != last.y).unwrap_or(true) {
        pts.push(last);
    }
    pts
}

fn dist_up_to(pts: &[Point], index: usize) -> f64 {
    let mut d = 0.0;
    for i in 1..=index {
        let a = pts[i - 1];
        let b = pts[i];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        d += (dx * dx + dy * dy).sqrt();
    }
    d
}

fn within(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}
