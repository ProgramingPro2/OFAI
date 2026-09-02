//! Tick implementations for later-system executions.

use std::collections::{HashMap, HashSet};

use crate::config::*;
use crate::game::Game;
use crate::js_math::js_exp;
use crate::later::{has_factory_nearby, LaterExec};
use crate::map::TileRef;
use crate::parabola::ParabolaPath;
use crate::path::{PathStatus, PathStepper};
use crate::rail::is_trade_type;
use crate::rng::PseudoRandom;
use crate::units::{UnitType, WarshipMode};

pub fn tick_later(game: &mut Game, idx: usize, ticks: u32) {
    let dummy = LaterExec::RailRecompute { active: false };
    let Some(exec) = game.swap_later(idx, dummy) else {
        return;
    };
    let exec = tick_one(game, exec, ticks);
    let _ = game.swap_later(idx, exec);
}

pub(crate) fn tick_one(game: &mut Game, exec: LaterExec, ticks: u32) -> LaterExec {
    match exec {
        LaterExec::Port {
            mut active,
            unit_id,
            mut rng,
            mut check_offset,
            mut rejections,
            mut inited,
        } => {
            if !inited {
                rng = Some(PseudoRandom::new(ticks as i32));
                check_offset = ticks % 10;
                inited = true;
            }
            let Some(u) = game.units().iter().find(|u| u.id == unit_id).cloned() else {
                return LaterExec::Port {
                    active: false,
                    unit_id,
                    rng,
                    check_offset,
                    rejections,
                    inited,
                };
            };
            if !u.active {
                active = false;
            } else if !u.under_construction {
                if !u.train_station && has_factory_nearby(game, u.tile) {
                    game.enqueue_later(LaterExec::TrainStation {
                        active: true,
                        unit_id,
                        spawn_trains: false,
                        station_id: None,
                        rng: None,
                        last_spawn: 0,
                        inited: false,
                    });
                    if let Some(un) = game.units_mut().iter_mut().find(|x| x.id == unit_id) {
                        un.train_station = true;
                    }
                }
                if (game.ticks() + check_offset) % 10 == 0 {
                    if let Some(r) = rng.as_mut() {
                        let nships = game
                            .units()
                            .iter()
                            .filter(|x| x.active && x.ty == UnitType::TradeShip)
                            .count() as u32;
                        let mut spawn = false;
                        for _ in 0..u.level {
                            let rate = trade_ship_spawn_rate(rejections, nships);
                            if r.chance(rate) {
                                rejections = 0;
                                spawn = true;
                                break;
                            }
                            rejections += 1;
                        }
                        if spawn {
                            if let Some(dst) = pick_trade_port(game, r, &u) {
                                game.enqueue_later(LaterExec::TradeShip {
                                    active: true,
                                    owner: u.owner_id.clone(),
                                    src_port: unit_id,
                                    dst_port: dst,
                                    unit_id: None,
                                    tiles_traveled: 0,
                                    last_move: ticks,
                                    stepper: PathStepper::new(),
                                    inited: false,
                                    was_captured: false,
                                });
                            }
                        }
                    }
                }
            }
            LaterExec::Port {
                active,
                unit_id,
                rng,
                check_offset,
                rejections,
                inited,
            }
        }
        LaterExec::Factory {
            mut active,
            unit_id,
            mut station_created,
        } => {
            if !station_created {
                create_factory_stations(game, unit_id);
                station_created = true;
            }
            let still = game.units().iter().any(|u| u.id == unit_id && u.active);
            if !still {
                active = false;
            }
            LaterExec::Factory {
                active,
                unit_id,
                station_created,
            }
        }
        LaterExec::City {
            mut active,
            unit_id,
            mut station_created,
        } => {
            if !station_created {
                if let Some(u) = game.units().iter().find(|u| u.id == unit_id).cloned() {
                    if has_factory_nearby(game, u.tile) {
                        game.enqueue_later(LaterExec::TrainStation {
                            active: true,
                            unit_id,
                            spawn_trains: false,
                            station_id: None,
                            rng: None,
                            last_spawn: 0,
                            inited: false,
                        });
                        if let Some(un) = game.units_mut().iter_mut().find(|x| x.id == unit_id) {
                            un.train_station = true;
                        }
                    }
                }
                station_created = true;
            }
            if !game.units().iter().any(|u| u.id == unit_id && u.active) {
                active = false;
            }
            LaterExec::City {
                active,
                unit_id,
                station_created,
            }
        }
        LaterExec::TrainStation {
            mut active,
            unit_id,
            spawn_trains,
            mut station_id,
            mut rng,
            mut last_spawn,
            mut inited,
        } => {
            if !inited {
                if spawn_trains {
                    rng = Some(PseudoRandom::new(ticks as i32));
                }
                inited = true;
            }
            if station_id.is_none() {
                if let Some(u) = game.units().iter().find(|u| u.id == unit_id).cloned() {
                    let nearby = nearby_station_units(game, u.tile);
                    let sid = game.later.rail.connect_station(
                        &game.map,
                        &game.mini_map,
                        unit_id,
                        u.tile,
                        &nearby,
                        is_trade_type(u.ty),
                    );
                    station_id = Some(sid);
                }
            }
            if let Some(sid) = station_id {
                if !game.units().iter().any(|u| u.id == unit_id && u.active) {
                    active = false;
                } else if spawn_trains {
                    maybe_spawn_train(game, unit_id, sid, &mut rng, &mut last_spawn, ticks);
                }
            }
            LaterExec::TrainStation {
                active,
                unit_id,
                spawn_trains,
                station_id,
                rng,
                last_spawn,
                inited,
            }
        }
        LaterExec::Train {
            mut active,
            owner,
            source,
            dest,
            mut engine,
            mut cars,
            mut stations,
            mut tiles,
            mut used,
            mut current,
            mut trade_stops,
            inited,
        } => {
            if active && inited {
                if let Some(eid) = engine {
                    if !game.units().iter().any(|u| u.id == eid && u.active)
                        || !train_source_or_dest_active(game, &stations)
                    {
                        delete_train(game, engine, &cars);
                        active = false;
                    } else if let Some(next) = train_next_tile(
                        game,
                        &owner,
                        &mut stations,
                        &mut tiles,
                        &mut used,
                        &mut current,
                        &mut trade_stops,
                        cars.len(),
                    ) {
                        // Official TrainExecution: `if (tile)` is JS-falsy for tile 0,
                        // so a railroad that lands on tile 0 deletes without dest gold.
                        if next == 0 {
                            delete_train(game, engine, &cars);
                            active = false;
                        } else {
                            update_cars(game, &cars, &used);
                            game.move_unit(eid, next);
                        }
                    } else {
                        delete_train(game, engine, &cars);
                        active = false;
                    }
                }
            }
            LaterExec::Train {
                active,
                owner,
                source,
                dest,
                engine,
                cars,
                stations,
                tiles,
                used,
                current,
                trade_stops,
                inited,
            }
        }
        LaterExec::TradeShip {
            mut active,
            owner,
            src_port,
            mut dst_port,
            mut unit_id,
            mut tiles_traveled,
            last_move,
            mut stepper,
            mut inited,
            mut was_captured,
        } => {
            if !inited {
                inited = true;
            }
            if unit_id.is_none() {
                if let Some(src) = game.units().iter().find(|u| u.id == src_port).map(|u| u.tile) {
                    if let Some(id) = game.build_unit(&owner, UnitType::TradeShip, src, 0.0) {
                        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                            u.target_unit = Some(dst_port);
                            u.last_safe_pirates = ticks;
                        }
                        unit_id = Some(id);
                    } else {
                        active = false;
                    }
                } else {
                    active = false;
                }
            }
            if let Some(id) = unit_id {
                if !game.units().iter().any(|u| u.id == id && u.active) {
                    active = false;
                } else {
                    let ship_owner = game
                        .units()
                        .iter()
                        .find(|u| u.id == id)
                        .map(|u| u.owner_id.clone());
                    if !was_captured && ship_owner.as_ref() != Some(&owner) {
                        was_captured = true;
                    }
                    let cur = game.units().iter().find(|u| u.id == id).map(|u| u.tile).unwrap();
                    let src_owner = game
                        .units()
                        .iter()
                        .find(|u| u.id == src_port)
                        .map(|u| u.owner_id.clone());
                    let dst_owner = game
                        .units()
                        .iter()
                        .find(|u| u.id == dst_port)
                        .map(|u| u.owner_id.clone());
                    let dst_active = game
                        .units()
                        .iter()
                        .any(|u| u.id == dst_port && u.active);
                    let same_port_owner = src_owner.is_some() && src_owner == dst_owner;
                    let ship_cannot_trade = !was_captured
                        && ship_owner.as_ref().zip(dst_owner.as_ref()).is_some_and(
                            |(so, dso)| so == dso || !game.can_trade(so, dso),
                        );
                    if same_port_owner
                        || (!was_captured && (!dst_active || ship_cannot_trade))
                    {
                        game.delete_unit(id);
                        active = false;
                    } else {
                        if was_captured
                            && (!dst_active
                                || ship_owner.as_ref() != dst_owner.as_ref())
                        {
                            match nearest_captured_trade_port(game, ship_owner.as_deref(), cur)
                            {
                                Some(new_dst) => {
                                    dst_port = new_dst;
                                    if let Some(u) =
                                        game.units_mut().iter_mut().find(|u| u.id == id)
                                    {
                                        u.target_unit = Some(new_dst);
                                    }
                                }
                                None => {
                                    game.delete_unit(id);
                                    active = false;
                                }
                            }
                        }
                        if active {
                            let Some(dst_tile) = game
                                .units()
                                .iter()
                                .find(|u| u.id == dst_port && u.active)
                                .map(|u| u.tile)
                            else {
                                game.delete_unit(id);
                                active = false;
                                return LaterExec::TradeShip {
                                    active,
                                    owner,
                                    src_port,
                                    dst_port,
                                    unit_id,
                                    tiles_traveled,
                                    last_move,
                                    stepper,
                                    inited,
                                    was_captured,
                                };
                            };
                            let src_owner = game
                                .units()
                                .iter()
                                .find(|u| u.id == src_port)
                                .map(|u| u.owner_id.clone());
                            let dst_owner = game
                                .units()
                                .iter()
                                .find(|u| u.id == dst_port && u.active)
                                .map(|u| u.owner_id.clone());
                            let (st, node) = game.water_step_with(&mut stepper, cur, dst_tile);
                            match st {
                                PathStatus::Complete => {
                                    let gold = trade_ship_gold(tiles_traveled);
                                    if was_captured {
                                        if let Some(so) = ship_owner.as_deref() {
                                            game.add_gold(so, gold);
                                        }
                                    } else if let (Some(so), Some(dso)) = (src_owner, dst_owner)
                                    {
                                        game.add_gold(&so, gold);
                                        game.add_gold(&dso, gold);
                                    }
                                    game.delete_unit(id);
                                    active = false;
                                }
                                PathStatus::Next => {
                                    if let Some(n) = node {
                                        if game.map.is_water(n) && game.map.is_shoreline(n) {
                                            if let Some(u) =
                                                game.units_mut().iter_mut().find(|u| u.id == id)
                                            {
                                                u.last_safe_pirates = ticks;
                                            }
                                        }
                                        game.move_unit(id, n);
                                        tiles_traveled += 1;
                                    }
                                }
                                PathStatus::NotFound => {
                                    game.delete_unit(id);
                                    active = false;
                                }
                            }
                        }
                    }
                }
            }
            LaterExec::TradeShip {
                active,
                owner,
                src_port,
                dst_port,
                unit_id,
                tiles_traveled,
                last_move,
                stepper,
                inited,
                was_captured,
            }
        }
        LaterExec::Warship {
            mut active,
            owner,
            patrol,
            mut unit_id,
            mut rng,
            mut last_shell,
            last_move,
            mut stepper,
            mut already_sent,
            mut inited,
            mut heal_rem,
            last_manual,
        } => {
            if rng.is_none() {
                rng = Some(PseudoRandom::new(ticks as i32));
            }
            if !inited {
                inited = true;
                if let Some(spawn) = game.can_build(&owner, UnitType::Warship, patrol) {
                    if let Some(id) = game.build_unit(&owner, UnitType::Warship, spawn, 0.0) {
                        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                            u.patrol_tile = Some(patrol);
                            u.target_tile = None;
                        }
                        unit_id = Some(id);
                    } else {
                        active = false;
                    }
                } else {
                    active = false;
                }
            }
            if let Some(id) = unit_id {
                let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
                    active = false;
                    return LaterExec::Warship {
                        active,
                        owner,
                        patrol,
                        unit_id,
                        rng,
                        last_shell,
                        last_move,
                        stepper,
                        already_sent,
                        inited,
                        heal_rem,
                        last_manual,
                    };
                };
                if ship.health <= 0 || !ship.active {
                    game.delete_unit(id);
                    active = false;
                } else {
                    let health_before = ship.health;
                    heal_warship(game, id);
                    let doomed = game.player(&owner).map(|p| p.in_doomsday).unwrap_or(false);
                    if doomed {
                        if let Some(u) = game.units().iter().find(|x| x.id == id) {
                            if u.warship_state != WarshipMode::Patrolling {
                                cancel_repair_retreat(game, id, &mut heal_rem, true);
                            }
                        }
                    }
                    let docked = game
                        .units()
                        .iter()
                        .find(|u| u.id == id)
                        .map(|u| u.warship_state == WarshipMode::Docked)
                        .unwrap_or(false);
                    // Official `healWarship` applies docked active healing before
                    // `isFullyHealed` / `cancelRepairRetreat`. Applying it after
                    // that check keeps the ship docked one extra tick after it
                    // reaches max health (warship 219 at t=7525).
                    if docked && !doomed {
                        apply_docked_healing(game, id, &mut heal_rem);
                    }
                    if docked {
                        let port_gone = current_retreat_port(game, id).is_none();
                        let full = game
                            .units()
                            .iter()
                            .find(|u| u.id == id)
                            .map(|u| u.health >= u.max_health)
                            .unwrap_or(true);
                        if port_gone || full {
                            cancel_repair_retreat(game, id, &mut heal_rem, true);
                        }
                        if !game
                            .units()
                            .iter()
                            .find(|u| u.id == id)
                            .map(|u| u.warship_state == WarshipMode::Docked)
                            .unwrap_or(false)
                        {
                            tick_warship_combat_or_patrol(
                                game,
                                &owner,
                                id,
                                patrol,
                                ticks,
                                &mut last_shell,
                                &mut already_sent,
                                rng.as_mut(),
                                &mut stepper,
                            );
                        }
                    } else if handle_repair_retreat(
                        game,
                        &owner,
                        id,
                        ticks,
                        &mut last_shell,
                        &mut already_sent,
                        &mut heal_rem,
                        &mut stepper,
                    ) {
                    } else if should_start_repair_retreat(game, id, health_before, ticks, last_manual)
                    {
                        start_repair_retreat(game, id, &mut heal_rem);
                        let _ = handle_repair_retreat(
                            game,
                            &owner,
                            id,
                            ticks,
                            &mut last_shell,
                            &mut already_sent,
                            &mut heal_rem,
                            &mut stepper,
                        );
                    } else {
                        tick_warship_combat_or_patrol(
                            game,
                            &owner,
                            id,
                            patrol,
                            ticks,
                            &mut last_shell,
                            &mut already_sent,
                            rng.as_mut(),
                            &mut stepper,
                        );
                    }
                }
            }
            LaterExec::Warship {
                active,
                owner,
                patrol,
                unit_id,
                rng,
                last_shell,
                last_move,
                stepper,
                already_sent,
                inited,
                heal_rem,
                last_manual,
            }
        }
        LaterExec::Nuke {
            mut active,
            owner,
            ty,
            dst,
            mut src,
            mut unit_id,
            speed,
            mut wait,
            up,
            mut path,
            mut inited,
        } => {
            if !inited {
                inited = true;
            }
            if unit_id.is_none() {
                if let Some(spawn) = game.can_build(&owner, ty, dst) {
                    src = Some(src.unwrap_or(spawn));
                    let from = src.unwrap();
                    if let Some(id) = game.build_unit(&owner, ty, from, 0.0) {
                        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                            u.target_tile = Some(dst);
                        }
                        if let Some(silo) = game.units_mut().iter_mut().find(|u| {
                            u.active
                                && u.ty == UnitType::MissileSilo
                                && u.owner_id == owner
                                && u.tile == spawn
                        }) {
                            let mut last = 0u32;
                            for &t in &silo.missile_queue {
                                last = last.max(t + 1).max(last + 1);
                            }
                            if last > ticks {
                                wait += last - ticks;
                            }
                            silo.missile_queue.push(ticks);
                        }
                        path = Some(ParabolaPath::new(&game.map, from, dst, speed, up));
                        if let Some(p) = path.as_ref() {
                            let tiles = p.all_tiles(&game.map);
                            if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                                u.trajectory = tiles;
                            }
                        }
                        unit_id = Some(id);
                        maybe_break_nuke_alliances(game, &owner, dst, ty);
                    } else {
                        active = false;
                    }
                } else {
                    active = false;
                }
            } else if let Some(id) = unit_id {
                if !game.units().iter().any(|u| u.id == id && u.active) {
                    active = false;
                } else if wait > 0 {
                    wait -= 1;
                } else if let Some(p) = path.as_mut() {
                    match p.increment(speed as f64) {
                        None => {
                            game.move_unit(id, dst);
                            // Official NukeExecution: if a SAM missile targeting this
                            // nuke is within defaultSamMissileSpeed of the dest, skip
                            // detonation so the SAM remains the source of truth.
                            if !sam_missile_blocks_detonate(game, id, dst, &owner) {
                                detonate(game, id, ty, dst);
                                active = false;
                            }
                        }
                        Some((x, y)) => {
                            if let Ok(tile) = game.map.ref_xy(x.floor() as u32, y.floor() as u32) {
                                let tidx = p.current_index();
                                let targetable = nuke_tile_targetable(game, src, dst, tile);
                                game.move_unit(id, tile);
                                if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                                    u.trajectory_index = tidx;
                                    u.targetable = targetable;
                                }
                            }
                        }
                    }
                }
            }
            LaterExec::Nuke {
                active,
                owner,
                ty,
                dst,
                src,
                unit_id,
                speed,
                wait,
                up,
                path,
                inited,
            }
        }
        LaterExec::Mirv {
            mut active,
            owner,
            dst,
            mut unit_id,
            mut rng,
            mut path,
            mut spawned,
            mut inited,
        } => {
            if rng.is_none() {
                rng = Some(PseudoRandom::new(
                    ticks as i32 + crate::rng::simple_hash(&owner),
                ));
            }
            if !inited {
                // Official MirvExecution.init: break alliance and pin relations at -100.
                if let Some(tid) =
                    game.player_by_small(game.map.owner_id(dst)).map(|p| p.id().to_string())
                {
                    if tid != owner {
                        if let Some(al) = game
                            .alliances()
                            .iter()
                            .find(|a| a.involves(&owner, &tid))
                            .cloned()
                        {
                            let t = game.ticks();
                            game.break_alliance(&al, &owner, t);
                        }
                        if let Some(p) = game.player_mut(&tid) {
                            p.update_relation(&owner, -100.0);
                        }
                        if let Some(p) = game.player_mut(&owner) {
                            p.update_relation(&tid, -100.0);
                        }
                    }
                }
                inited = true;
            }
            if !spawned {
                if let Some(spawn) = game.can_build(&owner, UnitType::Mirv, dst) {
                    if let Some(id) = game.build_unit(&owner, UnitType::Mirv, spawn, 0.0) {
                        game.later.num_mirvs += 1;
                        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                            u.target_tile = Some(dst);
                        }
                        if let Some(silo) = game.units_mut().iter_mut().find(|u| {
                            u.active
                                && u.ty == UnitType::MissileSilo
                                && u.owner_id == owner
                                && u.tile == spawn
                        }) {
                            silo.missile_queue.push(ticks);
                        }
                        let sx = game.map.x(spawn) as i32;
                        let dx = game.map.x(dst) as i32;
                        let dy = game.map.y(dst) as i32;
                        let sep_x = (dx + sx) / 2;
                        let sep_y = (dy - 500).max(0) + 50;
                        let sep = game
                            .map
                            .ref_xy(sep_x as u32, sep_y as u32)
                            .unwrap_or(dst);
                        path = Some(ParabolaPath::new(
                            &game.map,
                            spawn,
                            sep,
                            nuke_speed(UnitType::Mirv),
                            true,
                        ));
                        unit_id = Some(id);
                        spawned = true;
                    } else {
                        active = false;
                    }
                } else {
                    active = false;
                }
            }
            if let Some(id) = unit_id {
                if !game.units().iter().any(|u| u.id == id && u.active) {
                    active = false;
                } else if let Some(p) = path.as_mut() {
                    // Official MirvExecution builds and advances the parabola on the same tick.
                    match p.increment(nuke_speed(UnitType::Mirv) as f64) {
                        None => {
                            let src = game
                                .units()
                                .iter()
                                .find(|u| u.id == id)
                                .map(|u| u.tile)
                                .unwrap_or(dst);
                            drop_mirv_warheads(game, &owner, dst, src, rng.as_mut());
                            game.delete_unit(id);
                            active = false;
                        }
                        Some((x, y)) => {
                            if let Ok(tile) = game.map.ref_xy(x.floor() as u32, y.floor() as u32) {
                                game.move_unit(id, tile);
                            }
                        }
                    }
                }
            }
            LaterExec::Mirv {
                active,
                owner,
                dst,
                unit_id,
                rng,
                path,
                spawned,
                inited,
            }
        }
        LaterExec::Silo { mut active, unit_id } => {
            if let Some(u) = game.units().iter().find(|u| u.id == unit_id).cloned() {
                if u.under_construction {
                    // wait
                } else if !u.active {
                    active = false;
                } else if let Some(&front) = u.missile_queue.first() {
                    if SILO_COOLDOWN.saturating_sub(ticks.saturating_sub(front)) == 0 {
                        if let Some(s) = game.units_mut().iter_mut().find(|x| x.id == unit_id) {
                            if !s.missile_queue.is_empty() {
                                s.missile_queue.remove(0);
                            }
                        }
                    }
                }
            } else {
                active = false;
            }
            LaterExec::Silo { active, unit_id }
        }
        LaterExec::Sam {
            mut active,
            owner,
            unit_id,
            mut rng,
            mut last_fire,
            mut inited,
        } => {
            if !inited {
                rng = Some(PseudoRandom::new(ticks as i32));
                inited = true;
            }
            let Some(sam) = game.units().iter().find(|u| u.id == unit_id).cloned() else {
                return LaterExec::Sam {
                    active: false,
                    owner,
                    unit_id,
                    rng,
                    last_fire,
                    inited,
                };
            };
            if !sam.active {
                active = false;
            } else if !sam.under_construction {
                // Official SAMLauncherExecution reloads before targeting so a
                // cooldown that expires this tick can fire immediately.
                if let Some(&front) = sam.missile_queue.first() {
                    if SAM_COOLDOWN.saturating_sub(ticks.saturating_sub(front)) == 0 {
                        if let Some(s) = game.units_mut().iter_mut().find(|u| u.id == unit_id) {
                            if !s.missile_queue.is_empty() {
                                s.missile_queue.remove(0);
                            }
                        }
                    }
                }
                let cooling = game
                    .units()
                    .iter()
                    .find(|u| u.id == unit_id)
                    .map(|s| s.missile_queue.len() >= s.level as usize)
                    .unwrap_or(true);
                if !cooling {
                    let targets = sam_valid_targets(game, &sam, ticks);
                    for (nid, tile) in targets {
                        if game
                            .units()
                            .iter()
                            .find(|u| u.id == unit_id)
                            .map(|s| s.missile_queue.len() >= s.level as usize)
                            .unwrap_or(true)
                        {
                            break;
                        }
                        last_fire = ticks;
                        if let Some(s) = game.units_mut().iter_mut().find(|u| u.id == unit_id) {
                            s.missile_queue.push(ticks);
                        }
                        if let Some(n) = game.units_mut().iter_mut().find(|u| u.id == nid) {
                            n.targeted_by_sam = true;
                        }
                        game.enqueue_later(LaterExec::SamMissile {
                            active: true,
                            unit_id: None,
                            owner: owner.clone(),
                            sam: unit_id,
                            target: nid,
                            dst: tile,
                            last_move: ticks,
                            air_seed: 0,
                            air_path: None,
                            air_index: 0,
                            air_last_to: None,
                            inited: false,
                        });
                    }
                }
            }
            LaterExec::Sam {
                active,
                owner,
                unit_id,
                rng,
                last_fire,
                inited,
            }
        }
        LaterExec::SamMissile {
            mut active,
            mut unit_id,
            owner,
            sam,
            target,
            dst,
            last_move,
            mut air_seed,
            mut air_path,
            mut air_index,
            mut air_last_to,
            mut inited,
        } => {
            if !inited {
                inited = true;
                air_seed = ticks as i32;
                if let Some(stile) = game.units().iter().find(|u| u.id == sam).map(|u| u.tile) {
                    if let Some(id) = game.build_unit(&owner, UnitType::SamMissile, stile, 0.0) {
                        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                            u.target_unit = Some(target);
                        }
                        unit_id = Some(id);
                    } else {
                        active = false;
                    }
                } else {
                    active = false;
                }
            }
            if let Some(id) = unit_id {
                let missile_ok = game.units().iter().any(|u| u.id == id && u.active);
                if !missile_ok {
                    active = false;
                } else {
                    let tgt = game.units().iter().find(|u| u.id == target).cloned();
                    let sam_ok = game.units().iter().any(|u| u.id == sam && u.active);
                    let tgt_ok = tgt.as_ref().map(|t| t.active).unwrap_or(false);
                    let same_owner = tgt.as_ref().map(|t| t.owner_id == owner).unwrap_or(false);
                    let ty_ok = tgt.as_ref().map(|t| {
                        matches!(
                            t.ty,
                            UnitType::AtomBomb | UnitType::HydrogenBomb | UnitType::MirvWarhead
                        )
                    }).unwrap_or(false);
                    if !tgt_ok || !sam_ok || same_owner || !ty_ok {
                        if tgt_ok {
                            if let Some(n) = game.units_mut().iter_mut().find(|u| u.id == target) {
                                n.targeted_by_sam = false;
                            }
                        }
                        game.delete_unit(id);
                        active = false;
                    } else {
                        // Official SAMMissileExecution uses PathFinding.Air and
                        // intercepts on PathStatus.COMPLETE, not on arriving at dst.
                        for _ in 0..DEFAULT_SAM_MISSILE_SPEED {
                            let cur = game
                                .units()
                                .iter()
                                .find(|u| u.id == id)
                                .map(|u| u.tile)
                                .unwrap();
                            let (st, node) = air_stepper_next(
                                game,
                                cur,
                                dst,
                                air_seed,
                                &mut air_path,
                                &mut air_index,
                                &mut air_last_to,
                            );
                            match st {
                                PathStatus::Complete => {
                                    game.delete_unit_by(target, Some(owner.as_str()));
                                    game.delete_unit(id);
                                    active = false;
                                    break;
                                }
                                PathStatus::Next => {
                                    if let Some(n) = node {
                                        game.move_unit(id, n);
                                    }
                                }
                                PathStatus::NotFound => {
                                    game.delete_unit(id);
                                    active = false;
                                    break;
                                }
                            }
                            if !active {
                                break;
                            }
                        }
                    }
                }
            }
            LaterExec::SamMissile {
                active,
                unit_id,
                owner,
                sam,
                target,
                dst,
                last_move,
                air_seed,
                air_path,
                air_index,
                air_last_to,
                inited,
            }
        }
        LaterExec::Shell {
            mut active,
            owner,
            src,
            shooter,
            target,
            mut unit_id,
            last_move,
            air_seed,
            mut air_path,
            mut air_index,
            mut air_last_to,
            mut destroy_at,
            mut inited,
        } => {
            if !inited {
                inited = true;
                if let Some(id) = game.build_unit(&owner, UnitType::Shell, src, 0.0) {
                    unit_id = Some(id);
                } else {
                    active = false;
                }
            }
            if let Some(id) = unit_id {
                let shell_ok = game.units().iter().any(|u| u.id == id && u.active);
                if !shell_ok {
                    active = false;
                    return LaterExec::Shell {
                        active,
                        owner,
                        src,
                        shooter,
                        target,
                        unit_id,
                        last_move,
                        air_seed,
                        air_path,
                        air_index,
                        air_last_to,
                        destroy_at,
                        inited,
                    };
                }
                let tgt = game.units().iter().find(|u| u.id == target).cloned();
                let shooter_active = game.units().iter().any(|u| u.id == shooter && u.active);
                let same_owner = tgt
                    .as_ref()
                    .map(|t| t.owner_id == owner)
                    .unwrap_or(false);
                if tgt.as_ref().map(|t| !t.active).unwrap_or(true)
                    || same_owner
                    || (destroy_at != -1 && ticks as i32 >= destroy_at)
                {
                    game.delete_unit(id);
                    active = false;
                    return LaterExec::Shell {
                        active,
                        owner,
                        src,
                        shooter,
                        target,
                        unit_id,
                        last_move,
                        air_seed,
                        air_path,
                        air_index,
                        air_last_to,
                        destroy_at,
                        inited,
                    };
                }
                if destroy_at == -1 && !shooter_active {
                    destroy_at = ticks as i32 + SHELL_LIFETIME as i32;
                }
                let goal = tgt.as_ref().unwrap().tile;
                for _ in 0..3 {
                    let cur = game.units().iter().find(|u| u.id == id).map(|u| u.tile).unwrap();
                    let (st, node) = air_stepper_next(
                        game,
                        cur,
                        goal,
                        air_seed,
                        &mut air_path,
                        &mut air_index,
                        &mut air_last_to,
                    );
                    match st {
                        PathStatus::Complete => {
                            apply_shell_damage(game, target, air_seed, &owner, shooter);
                            game.delete_unit(id);
                            active = false;
                            break;
                        }
                        PathStatus::Next => {
                            if let Some(n) = node {
                                game.move_unit(id, n);
                            }
                        }
                        PathStatus::NotFound => {
                            game.delete_unit(id);
                            active = false;
                            break;
                        }
                    }
                    if !active {
                        break;
                    }
                }
            }
            LaterExec::Shell {
                active,
                owner,
                src,
                shooter,
                target,
                unit_id,
                last_move,
                air_seed,
                air_path,
                air_index,
                air_last_to,
                destroy_at,
                inited,
            }
        }
        LaterExec::RailRecompute { active } => {
            game.later.rail.recompute();
            LaterExec::RailRecompute { active }
        }
        LaterExec::Doomsday { active } => {
            crate::doomsday::tick(game, ticks);
            LaterExec::Doomsday { active }
        }
        LaterExec::Upgrade {
            mut active,
            owner,
            unit_id,
            mut inited,
        } => {
            if !inited {
                inited = true;
                if let Some(ty) = game.units().iter().find(|u| u.id == unit_id).map(|u| u.ty) {
                    game.upgrade_structure_unit(&owner, unit_id, ty);
                }
                active = false;
            }
            LaterExec::Upgrade {
                active,
                owner,
                unit_id,
                inited,
            }
        }
    }
}

impl Game {
    pub(crate) fn units_mut(&mut self) -> &mut Vec<crate::units::Unit> {
        &mut self.units
    }

    pub(crate) fn map_ref(&self) -> &crate::map::GameMap {
        &self.map
    }

    fn mini_tile_of(&self, tile: TileRef) -> Option<TileRef> {
        self.mini_map
            .ref_xy(self.map.x(tile) / 2, self.map.y(tile) / 2)
            .ok()
    }

    /// Official `WaterManager.getWaterComponent` (mini-map components).
    fn get_water_component(&self, tile: TileRef) -> Option<i32> {
        let mini = self.mini_tile_of(tile)?;
        if self.mini_map.is_water(mini) {
            return self.mini_water.of_tile(&self.mini_map, mini);
        }
        let mut nbuf = [0u32; 4];
        let n = self.mini_map.neighbors4(mini, &mut nbuf);
        for i in 0..n {
            if self.mini_map.is_water(nbuf[i]) {
                if let Some(c) = self.mini_water.of_tile(&self.mini_map, nbuf[i]) {
                    return Some(c);
                }
            }
        }
        for i in 0..n {
            let mut nbuf2 = [0u32; 4];
            let n2 = self.mini_map.neighbors4(nbuf[i], &mut nbuf2);
            for j in 0..n2 {
                if self.mini_map.is_water(nbuf2[j]) {
                    if let Some(c) = self.mini_water.of_tile(&self.mini_map, nbuf2[j]) {
                        return Some(c);
                    }
                }
            }
        }
        None
    }

    /// Official `WaterManager.hasWaterComponent`.
    fn has_water_component(&self, tile: TileRef, component: i32) -> bool {
        let Some(mini) = self.mini_tile_of(tile) else {
            return false;
        };
        if self.mini_map.is_water(mini)
            && self.mini_water.of_tile(&self.mini_map, mini) == Some(component)
        {
            return true;
        }
        let mut nbuf = [0u32; 4];
        let n = self.mini_map.neighbors4(mini, &mut nbuf);
        for i in 0..n {
            if self.mini_map.is_water(nbuf[i])
                && self.mini_water.of_tile(&self.mini_map, nbuf[i]) == Some(component)
            {
                return true;
            }
        }
        for i in 0..n {
            let mut nbuf2 = [0u32; 4];
            let n2 = self.mini_map.neighbors4(nbuf[i], &mut nbuf2);
            for j in 0..n2 {
                if self.mini_map.is_water(nbuf2[j])
                    && self.mini_water.of_tile(&self.mini_map, nbuf2[j]) == Some(component)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Official TrainExecution.init builds units when the exec is inited,
    /// one tick before the first movement tick.
    pub(crate) fn init_train(&mut self, exec: &mut LaterExec) {
        let LaterExec::Train {
            active,
            owner,
            source,
            dest,
            engine,
            cars,
            stations,
            tiles,
            inited,
            ..
        } = exec
        else {
            return;
        };
        if *inited {
            return;
        }
        *stations = self.later.rail.find_station_path(&self.map, *source, *dest);
        if stations.len() <= 1 {
            *active = false;
        } else if let Some(seg) = self.later.rail.oriented_tiles(stations[0], stations[1]) {
            *tiles = seg;
            let spawn_tile = self
                .later
                .rail
                .stations
                .get(&stations[0])
                .map(|s| s.tile)
                .unwrap_or(0);
            if let Some(spawn) = self.can_build(owner, UnitType::Train, spawn_tile) {
                if let Some(eid) = self.build_unit(owner, UnitType::Train, spawn, 0.0) {
                    *engine = Some(eid);
                    if let Some(tail) = self.build_unit(owner, UnitType::Train, spawn, 0.0) {
                        cars.push(tail);
                    }
                    for _ in 0..5 {
                        if let Some(c) = self.build_unit(owner, UnitType::Train, spawn, 0.0) {
                            cars.push(c);
                        }
                    }
                } else {
                    *active = false;
                }
            } else {
                *active = false;
            }
        } else {
            *active = false;
        }
        *inited = true;
    }
}

fn nearby_station_units(game: &Game, tile: TileRef) -> Vec<(u32, TileRef, i64)> {
    game.later_grid_nearby(
        tile,
        TRAIN_STATION_MAX_RANGE,
        &[UnitType::City, UnitType::Port, UnitType::Factory],
    )
    .into_iter()
    .filter_map(|(id, d2)| {
        game.units()
            .iter()
            .find(|u| u.id == id)
            .map(|u| (u.id, u.tile, d2))
    })
    .collect()
}

fn create_factory_stations(game: &mut Game, factory_id: u32) {
    let Some(factory) = game.units().iter().find(|u| u.id == factory_id).cloned() else {
        return;
    };
    let nearby = game.later_grid_nearby(
        factory.tile,
        TRAIN_STATION_MAX_RANGE,
        &[UnitType::City, UnitType::Port, UnitType::Factory],
    );
    game.enqueue_later(LaterExec::TrainStation {
        active: true,
        unit_id: factory_id,
        spawn_trains: true,
        station_id: None,
        rng: None,
        last_spawn: 0,
        inited: false,
    });
    if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == factory_id) {
        u.train_station = true;
    }
    for (id, _) in nearby {
        if id == factory_id {
            continue;
        }
        if let Some(u) = game.units().iter().find(|u| u.id == id) {
            if u.train_station {
                continue;
            }
        }
        game.enqueue_later(LaterExec::TrainStation {
            active: true,
            unit_id: id,
            spawn_trains: false,
            station_id: None,
            rng: None,
            last_spawn: 0,
            inited: false,
        });
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
            u.train_station = true;
        }
    }
}

/// Official captured TradeShipExecution retarget: closest own port on
/// the same water component, skipping marked/under-construction.
fn nearest_captured_trade_port(game: &Game, ship_owner: Option<&str>, cur: TileRef) -> Option<u32> {
    let owner = ship_owner?;
    let component = game.get_water_component(cur)?;
    let mut best: Option<(u32, u32)> = None;
    for u in game.units() {
        if !u.active
            || u.ty != UnitType::Port
            || u.owner_id != owner
            || u.marked_for_deletion
            || u.under_construction
        {
            continue;
        }
        if !game.has_water_component(u.tile, component) {
            continue;
        }
        let d = game.map.manhattan(u.tile, cur);
        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, u.id));
        }
    }
    best.map(|(_, id)| id)
}

fn pick_trade_port(game: &Game, rng: &mut PseudoRandom, src: &crate::units::Unit) -> Option<u32> {
    let mut source_comps = HashSet::new();
    let mut nbuf = [0u32; 4];
    let n = game.map.neighbors4(src.tile, &mut nbuf);
    for i in 0..n {
        if game.map.is_water(nbuf[i]) {
            if let Some(c) = game.get_water_component(nbuf[i]) {
                source_comps.insert(c);
            }
        }
    }
    let mut ports: Vec<(u32, u32)> = Vec::new();
    // Official PortExecution.tradingPorts: alive players' current Port
    // units. Walk game.units by owner_id so tile-captured ports are
    // visible even if Player.units was stale.
    for u in game.units() {
        if !u.active || u.ty != UnitType::Port || u.owner_id == src.owner_id {
            continue;
        }
        let Some(owner) = game.player(&u.owner_id) else {
            continue;
        };
        if !owner.is_alive() || !game.can_trade(&src.owner_id, &u.owner_id) {
            continue;
        }
        if source_comps
            .iter()
            .any(|c| game.has_water_component(u.tile, *c))
        {
            ports.push((game.map.manhattan(src.tile, u.tile), u.id));
        }
    }
    ports.sort_by_key(|(d, _)| *d);
    if ports.is_empty() {
        return None;
    }
    let mut weighted = Vec::new();
    for (i, &(dist, id)) in ports.iter().enumerate() {
        let level = game
            .units()
            .iter()
            .find(|u| u.id == id)
            .map(|u| u.level)
            .unwrap_or(1);
        for _ in 0..level {
            weighted.push(id);
        }
        let too_close = dist < 300;
        let close_bonus = (i as f64) < proximity_bonus_ports(ports.len());
        if !too_close && close_bonus {
            for _ in 0..level {
                weighted.push(id);
            }
        }
        if !too_close {
            let other = game
                .units()
                .iter()
                .find(|u| u.id == id)
                .map(|u| u.owner_id.clone());
            if let Some(oid) = other {
                if game.is_friendly(
                    game.player(&src.owner_id).unwrap(),
                    game.player(&oid).unwrap(),
                ) {
                    for _ in 0..level {
                        weighted.push(id);
                    }
                }
            }
        }
    }
    if weighted.is_empty() {
        None
    } else {
        Some(rng.rand_element(&weighted))
    }
}

fn maybe_spawn_train(
    game: &mut Game,
    unit_id: u32,
    sid: u32,
    rng: &mut Option<PseudoRandom>,
    last_spawn: &mut u32,
    ticks: u32,
) {
    let Some(r) = rng.as_mut() else { return };
    if ticks < *last_spawn + 10 {
        return;
    }
    let Some(u) = game.units().iter().find(|x| x.id == unit_id).cloned() else {
        return;
    };
    let dests = game.later.rail.trade_stations(sid);
    let dests: Vec<u32> = dests
        .into_iter()
        .filter(|&d| {
            let Some(st) = game.later.rail.stations.get(&d) else {
                return false;
            };
            let Some(ou) = game.units().iter().find(|x| x.id == st.unit_id) else {
                return false;
            };
            game.can_trade(&u.owner_id, &ou.owner_id) || ou.owner_id == u.owner_id
        })
        .collect();
    if dests.is_empty() {
        return;
    }
    // Official Player.unitCount(Factory) sums factory levels, not buildings.
    let factories = game
        .units()
        .iter()
        .filter(|x| x.active && x.ty == UnitType::Factory && x.owner_id == u.owner_id)
        .map(|x| x.level)
        .sum::<u32>();
    let mut ok = false;
    for _ in 0..u.level {
        if r.chance(train_spawn_rate(factories)) {
            ok = true;
            break;
        }
    }
    if !ok {
        return;
    }
    let mut selected = None;
    let mut seen = 0i32;
    for &d in &dests {
        seen += 1;
        if r.next_int(0.0, seen as f64) == 0 {
            selected = Some(d);
        }
    }
    let Some(dest) = selected else { return };
    if dest == sid {
        return;
    }
    game.enqueue_later(LaterExec::Train {
        active: true,
        owner: u.owner_id,
        source: sid,
        dest,
        engine: None,
        cars: Vec::new(),
        stations: Vec::new(),
        tiles: Vec::new(),
        used: Vec::new(),
        current: 0,
        trade_stops: 0,
        inited: false,
    });
    *last_spawn = ticks;
}

fn train_next_tile(
    game: &mut Game,
    owner: &str,
    stations: &mut Vec<u32>,
    tiles: &mut Vec<TileRef>,
    used: &mut Vec<TileRef>,
    current: &mut i32,
    trade_stops: &mut u32,
    n_cars: usize,
) -> Option<TileRef> {
    if stations.len() < 2 || tiles.is_empty() {
        return None;
    }
    // Official TrainExecution.getNextTile: abort when the next station
    // cannot trade with the train owner (embargo after spawn).
    if !train_can_trade_dest(game, owner, stations) {
        return None;
    }
    let speed = 2i32;
    save_train_tiles(tiles, used, *current, speed, n_cars);
    *current += speed;
    let leftover = *current - tiles.len() as i32;
    if leftover >= 0 {
        if let Some(sid) = stations.get(1).copied() {
            if let Some(st) = game.later.rail.stations.get(&sid) {
                if let Some(u) = game.units().iter().find(|x| x.id == st.unit_id).cloned() {
                    if is_trade_type(u.ty) {
                        let rel = if u.owner_id == owner {
                            "self"
                        } else if game.is_allied(owner, &u.owner_id) {
                            "ally"
                        } else {
                            "other"
                        };
                        let gold = train_gold(rel, *trade_stops);
                        game.add_gold(owner, gold);
                        if u.owner_id != owner {
                            game.add_gold(&u.owner_id, gold);
                        }
                        *trade_stops += 1;
                    }
                }
            }
        }
        if stations.len() > 2 {
            stations.remove(0);
            if let Some(seg) = game.later.rail.oriented_tiles(stations[0], stations[1]) {
                *tiles = seg;
                *current = leftover;
                save_train_tiles(tiles, used, 0, leftover, n_cars);
                return tiles.get(*current as usize).copied();
            }
        }
        return None;
    }
    tiles.get(*current as usize).copied()
}

fn station_unit_active(game: &Game, sid: u32) -> bool {
    game.later
        .rail
        .stations
        .get(&sid)
        .and_then(|st| game.units().iter().find(|u| u.id == st.unit_id && u.active))
        .is_some()
}

fn train_source_or_dest_active(game: &Game, stations: &[u32]) -> bool {
    stations.len() > 1
        && station_unit_active(game, stations[0])
        && station_unit_active(game, stations[1])
}

fn train_can_trade_dest(game: &Game, owner: &str, stations: &[u32]) -> bool {
    if stations.len() < 2 {
        return false;
    }
    let Some(st) = game.later.rail.stations.get(&stations[1]) else {
        return false;
    };
    let Some(u) = game.units().iter().find(|x| x.id == st.unit_id) else {
        return false;
    };
    u.owner_id == owner || game.can_trade(&u.owner_id, owner)
}

fn save_train_tiles(
    tiles: &[TileRef],
    used: &mut Vec<TileRef>,
    from: i32,
    count: i32,
    n_cars: usize,
) {
    let cap = n_cars * 2 + 3;
    let mut tile_to_save = from;
    for _ in 0..count {
        if tile_to_save < 0 || tile_to_save as usize >= tiles.len() {
            break;
        }
        used.push(tiles[tile_to_save as usize]);
        if used.len() > cap {
            used.remove(0);
        }
        tile_to_save += 1;
    }
}

fn update_cars(game: &mut Game, cars: &[u32], used: &[TileRef]) {
    for (i, &cid) in cars.iter().enumerate().rev() {
        let idx = (i + 1) * 2 + 2;
        if used.len() > idx {
            game.move_unit(cid, used[idx]);
        }
    }
}

fn delete_train(game: &mut Game, engine: Option<u32>, cars: &[u32]) {
    if let Some(e) = engine {
        game.delete_unit(e);
    }
    for &c in cars {
        game.delete_unit(c);
    }
}

fn heal_warship(game: &mut Game, id: u32) {
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return;
    };
    if game
        .player(&ship.owner_id)
        .map(|p| p.in_doomsday)
        .unwrap_or(false)
    {
        return;
    }
    let r2 = (WARSHIP_PASSIVE_HEALING_RANGE as i64) * (WARSHIP_PASSIVE_HEALING_RANGE as i64);
    // Official `owner.units(Port)` walks Player._units insertion order.
    let near_port = owner_port_tiles(game, &ship.owner_id).iter().any(|&tile| {
        let dx = game.map.x(tile) as i64 - game.map.x(ship.tile) as i64;
        let dy = game.map.y(tile) as i64 - game.map.y(ship.tile) as i64;
        dx * dx + dy * dy <= r2
    });
    if near_port {
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
            u.health = (u.health + WARSHIP_PASSIVE_HEALING).min(u.max_health);
        }
    }
}

fn tick_warship_combat_or_patrol(
    game: &mut Game,
    owner: &str,
    id: u32,
    patrol: TileRef,
    ticks: u32,
    last_shell: &mut u32,
    already_sent: &mut Vec<u32>,
    rng: Option<&mut PseudoRandom>,
    stepper: &mut PathStepper,
) {
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return;
    };
    let tgt = find_best_warship_target(game, &ship, already_sent, true);
    if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
        u.target_unit = tgt;
    }
    let tgt_ty = tgt.and_then(|tid| game.units().iter().find(|u| u.id == tid).map(|u| u.ty));
    match tgt_ty {
        Some(UnitType::TransportShip) | Some(UnitType::Warship) => {
            shoot_warship_target(game, owner, id, tgt.unwrap(), ticks, last_shell, already_sent);
            if let Some(r) = rng {
                patrol_warship(game, id, patrol, r, stepper);
            }
        }
        Some(UnitType::TradeShip) => {
            hunt_trade_ship(game, id, tgt.unwrap(), stepper);
        }
        _ => {
            if let Some(r) = rng {
                patrol_warship(game, id, patrol, r, stepper);
            }
        }
    }
}

fn owner_port_tiles(game: &Game, owner: &str) -> Vec<TileRef> {
    let ids = game.player(owner).map(|p| p.units.clone()).unwrap_or_default();
    ids.iter()
        .filter_map(|&id| {
            game.units()
                .iter()
                .find(|u| u.id == id && u.ty == UnitType::Port)
                .map(|u| u.tile)
        })
        .collect()
}

fn euclid2(game: &Game, a: TileRef, b: TileRef) -> i64 {
    let dx = game.map.x(a) as i64 - game.map.x(b) as i64;
    let dy = game.map.y(a) as i64 - game.map.y(b) as i64;
    dx * dx + dy * dy
}

fn find_nearest_port(game: &Game, ship_id: u32) -> Option<TileRef> {
    let ship = game.units().iter().find(|u| u.id == ship_id)?;
    let wc = game.get_water_component(ship.tile)?;
    let mut best = None;
    let mut best_d = i64::MAX;
    for tile in owner_port_tiles(game, &ship.owner_id) {
        if !game.has_water_component(tile, wc) {
            continue;
        }
        let d = euclid2(game, ship.tile, tile);
        if d < best_d {
            best_d = d;
            best = Some(tile);
        }
    }
    best
}

fn should_start_repair_retreat(
    game: &Game,
    id: u32,
    health_before: i32,
    ticks: u32,
    last_manual: u32,
) -> bool {
    let Some(ship) = game.units().iter().find(|u| u.id == id) else {
        return false;
    };
    if ship.warship_state != WarshipMode::Patrolling {
        return false;
    }
    if game
        .player(&ship.owner_id)
        .map(|p| p.in_doomsday)
        .unwrap_or(false)
    {
        return false;
    }
    if ticks.saturating_sub(last_manual) < 50 {
        return false;
    }
    let thresh = ship.max_health * WARSHIP_RETREAT_HEALTH_PERCENT / 100;
    if health_before >= thresh {
        return false;
    }
    !owner_port_tiles(game, &ship.owner_id).is_empty()
}

fn start_repair_retreat(game: &mut Game, id: u32, heal_rem: &mut f64) {
    let Some(port) = find_nearest_port(game, id) else {
        return;
    };
    *heal_rem = 0.0;
    if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
        u.retreat_port = Some(port);
        u.warship_state = WarshipMode::Retreating;
        u.target_unit = None;
    }
}

fn cancel_repair_retreat(game: &mut Game, id: u32, heal_rem: &mut f64, clear_target: bool) {
    *heal_rem = 0.0;
    if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
        u.warship_state = WarshipMode::Patrolling;
        u.retreat_port = None;
        if clear_target {
            u.target_tile = None;
        }
    }
}

fn current_retreat_port(game: &Game, id: u32) -> Option<TileRef> {
    let ship = game.units().iter().find(|u| u.id == id)?;
    let port = ship.retreat_port?;
    owner_port_tiles(game, &ship.owner_id)
        .into_iter()
        .find(|t| *t == port)
}

fn docked_ships_at_port(game: &Game, owner: &str, port: TileRef, exclude: Option<u32>) -> Vec<u32> {
    game.later_grid_nearby(port, WARSHIP_DOCKING_RANGE, &[UnitType::Warship])
        .into_iter()
        .filter_map(|(sid, _)| {
            let u = game.units().iter().find(|x| x.id == sid)?;
            if exclude == Some(sid) {
                return None;
            }
            if u.owner_id != owner {
                return None;
            }
            if u.warship_state == WarshipMode::Patrolling {
                return None;
            }
            if u.target_tile.is_some() {
                return None;
            }
            Some(sid)
        })
        .collect()
}

fn is_port_full(game: &Game, owner: &str, port: TileRef, exclude: Option<u32>) -> bool {
    let level = game
        .units()
        .iter()
        .find(|u| u.ty == UnitType::Port && u.tile == port && u.owner_id == owner)
        .map(|u| u.level)
        .unwrap_or(1);
    docked_ships_at_port(game, owner, port, exclude).len() >= level as usize
}

fn nearest_available_port(game: &Game, ship_id: u32, exclude_self: bool) -> Option<(TileRef, i64)> {
    let ship = game.units().iter().find(|u| u.id == ship_id)?;
    let wc = game.get_water_component(ship.tile)?;
    let exclude = if exclude_self { Some(ship_id) } else { None };
    let mut best = None;
    let mut best_d = i64::MAX;
    for tile in owner_port_tiles(game, &ship.owner_id) {
        if is_port_full(game, &ship.owner_id, tile, exclude) {
            continue;
        }
        if !game.has_water_component(tile, wc) {
            continue;
        }
        let d = euclid2(game, ship.tile, tile);
        if d < best_d {
            best_d = d;
            best = Some(tile);
        }
    }
    best.map(|t| (t, best_d))
}

fn refresh_retreat_port(game: &mut Game, id: u32) -> bool {
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return false;
    };
    let ports = owner_port_tiles(game, &ship.owner_id);
    if ports.is_empty() {
        return false;
    }
    let current_ok = ship
        .retreat_port
        .is_some_and(|p| ports.iter().any(|t| *t == p));
    if !current_ok {
        let new_port = nearest_available_port(game, id, true).map(|(t, _)| t);
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
            u.retreat_port = new_port;
        }
        return new_port.is_some();
    }
    if let Some(cur) = ship.retreat_port {
        if is_port_full(game, &ship.owner_id, cur, None) {
            if let Some((alt, _)) = nearest_available_port(game, id, false) {
                if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                    u.retreat_port = Some(alt);
                }
            }
            return game
                .units()
                .iter()
                .find(|u| u.id == id)
                .and_then(|u| u.retreat_port)
                .is_some();
        }
        if let Some((closer, d2)) = nearest_available_port(game, id, false) {
            let cur_d = euclid2(game, ship.tile, cur) as f64;
            if (d2 as f64) < cur_d * WARSHIP_PORT_SWITCH && closer != cur {
                if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                    u.retreat_port = Some(closer);
                }
            }
        }
    }
    true
}

fn apply_docked_healing(game: &mut Game, id: u32, heal_rem: &mut f64) {
    let Some(port) = current_retreat_port(game, id) else {
        return;
    };
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return;
    };
    let docked = docked_ships_at_port(game, &ship.owner_id, port, None);
    if !docked.contains(&id) {
        return;
    }
    let level = game
        .units()
        .iter()
        .find(|u| u.ty == UnitType::Port && u.tile == port && u.owner_id == ship.owner_id)
        .map(|u| u.level)
        .unwrap_or(1);
    if docked.is_empty() {
        return;
    }
    let pool = level as f64 * WARSHIP_PORT_HEALING_BONUS as f64;
    *heal_rem += pool / docked.len() as f64;
    let add = heal_rem.floor() as i32;
    if add <= 0 {
        return;
    }
    *heal_rem -= add as f64;
    if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
        u.health = (u.health + add).min(u.max_health);
    }
}

fn handle_repair_retreat(
    game: &mut Game,
    owner: &str,
    id: u32,
    ticks: u32,
    last_shell: &mut u32,
    already_sent: &mut Vec<u32>,
    heal_rem: &mut f64,
    stepper: &mut PathStepper,
) -> bool {
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return false;
    };
    if ship.warship_state == WarshipMode::Patrolling {
        return false;
    }
    let aggro = find_best_warship_target(game, &ship, already_sent, false);
    if let Some(tid) = aggro {
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
            u.target_unit = Some(tid);
        }
        shoot_warship_target(game, owner, id, tid, ticks, last_shell, already_sent);
    }
    if !refresh_retreat_port(game, id) {
        cancel_repair_retreat(game, id, heal_rem, true);
        return false;
    }
    if aggro.is_none() {
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
            u.target_unit = None;
        }
    }
    let Some(port) = game
        .units()
        .iter()
        .find(|u| u.id == id)
        .and_then(|u| u.retreat_port)
    else {
        return false;
    };
    let dock_r2 = (WARSHIP_DOCKING_RANGE as i64) * (WARSHIP_DOCKING_RANGE as i64);
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return false;
    };
    if euclid2(game, ship.tile, port) <= dock_r2 {
        if !is_port_full(game, &ship.owner_id, port, Some(id)) {
            if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                u.target_tile = None;
                u.warship_state = WarshipMode::Docked;
            }
            return true;
        }
        if ship.health >= ship.max_health {
            cancel_repair_retreat(game, id, heal_rem, true);
            return false;
        }
        return true;
    }
    if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
        u.target_tile = Some(port);
    }
    let (st, node) = game.water_step_with(stepper, ship.tile, port);
    match st {
        PathStatus::Complete => {
            if let Some(n) = node {
                game.move_unit(id, n);
            }
            if node == Some(port) {
                if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                    u.target_tile = None;
                }
            }
        }
        PathStatus::Next => {
            if let Some(n) = node {
                game.move_unit(id, n);
            }
        }
        PathStatus::NotFound => {
            let new_port = nearest_available_port(game, id, true).map(|(t, _)| t);
            if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                u.retreat_port = new_port;
            }
            if new_port.is_none() {
                cancel_repair_retreat(game, id, heal_rem, true);
            }
        }
    }
    true
}

fn unit_has_health(ty: UnitType) -> bool {
    ty == UnitType::Warship
}

fn is_safe_from_pirates(game: &Game, u: &crate::units::Unit) -> bool {
    game.ticks().saturating_sub(u.last_safe_pirates) < SAFE_FROM_PIRATES_COOLDOWN
}

fn can_attack_ids(game: &Game, attacker: &str, target: &str) -> bool {
    let Some(a) = game.player(attacker) else {
        return false;
    };
    let Some(b) = game.player(target) else {
        return false;
    };
    game.can_attack_player(a, b)
}

fn dest_owner_friendly(game: &Game, dest_owner: &str, warship_owner: &str) -> bool {
    if dest_owner == warship_owner {
        return true;
    }
    let Some(a) = game.player(dest_owner) else {
        return false;
    };
    let Some(b) = game.player(warship_owner) else {
        return false;
    };
    game.is_friendly(a, b)
}

fn find_best_warship_target(
    game: &Game,
    ship: &crate::units::Unit,
    already_sent: &[u32],
    include_trade: bool,
) -> Option<u32> {
    let types = if include_trade {
        &[
            UnitType::TransportShip,
            UnitType::Warship,
            UnitType::TradeShip,
        ][..]
    } else {
        &[UnitType::TransportShip, UnitType::Warship][..]
    };
    let nearby = game.later_grid_nearby(ship.tile, WARSHIP_TARGETING_RANGE, types);
    let mut has_reachable_port = None;
    let mut patrol_range_sq = None;
    let mut best: Option<(u32, i32, i64)> = None;
    for (id, d2) in nearby {
        let Some(u) = game.units().iter().find(|x| x.id == id) else {
            continue;
        };
        if u.id == ship.id || u.owner_id == ship.owner_id {
            continue;
        }
        if !can_attack_ids(game, &ship.owner_id, &u.owner_id) {
            continue;
        }
        if already_sent.contains(&u.id) {
            continue;
        }
        if u.ty == UnitType::Warship && u.warship_state == WarshipMode::Docked {
            continue;
        }
        if include_trade && u.ty == UnitType::TradeShip {
            if has_reachable_port.is_none() {
                let wc = game.get_water_component(ship.tile);
                has_reachable_port = Some(wc.is_some() && {
                    let c = wc.unwrap();
                    game.units().iter().any(|p| {
                        p.active
                            && p.ty == UnitType::Port
                            && p.owner_id == ship.owner_id
                            && !p.marked_for_deletion
                            && !p.under_construction
                            && game.has_water_component(p.tile, c)
                    })
                });
                patrol_range_sq = Some((WARSHIP_PATROL_RANGE as i64) * (WARSHIP_PATROL_RANGE as i64));
            }
            let dest_owner = u.target_unit.and_then(|did| {
                game.units().iter().find(|x| x.id == did).map(|x| x.owner_id.clone())
            });
            if !has_reachable_port.unwrap()
                || ship.patrol_tile.is_none()
                || is_safe_from_pirates(game, u)
                || dest_owner
                    .as_deref()
                    .is_some_and(|d| dest_owner_friendly(game, d, &ship.owner_id))
            {
                continue;
            }
            if let Some(pt) = ship.patrol_tile {
                let dx = game.map.x(pt) as i64 - game.map.x(u.tile) as i64;
                let dy = game.map.y(pt) as i64 - game.map.y(u.tile) as i64;
                if dx * dx + dy * dy > patrol_range_sq.unwrap() {
                    continue;
                }
            }
        }
        let prio = match u.ty {
            UnitType::TransportShip => 0,
            UnitType::Warship => 1,
            _ => 2,
        };
        if best
            .map(|(_, bp, bd)| prio < bp || (prio == bp && d2 < bd))
            .unwrap_or(true)
        {
            best = Some((id, prio, d2));
        }
    }
    best.map(|(id, _, _)| id)
}

fn shoot_warship_target(
    game: &mut Game,
    owner: &str,
    shooter: u32,
    target: u32,
    ticks: u32,
    last_shell: &mut u32,
    already_sent: &mut Vec<u32>,
) {
    if ticks.saturating_sub(*last_shell) <= WARSHIP_SHELL_RATE {
        return;
    }
    let tgt_ty = game.units().iter().find(|u| u.id == target).map(|u| u.ty);
    if tgt_ty != Some(UnitType::TransportShip) {
        *last_shell = ticks;
    }
    let src = game
        .units()
        .iter()
        .find(|u| u.id == shooter)
        .map(|u| u.tile)
        .unwrap();
    game.enqueue_later(LaterExec::Shell {
        active: true,
        owner: owner.to_string(),
        src,
        shooter,
        target,
        unit_id: None,
        last_move: ticks,
        air_seed: ticks as i32,
        air_path: None,
        air_index: 0,
        air_last_to: None,
        destroy_at: -1,
        inited: false,
    });
    if tgt_ty.is_some_and(|ty| !unit_has_health(ty)) {
        already_sent.push(target);
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == shooter) {
            u.target_unit = None;
        }
    }
}

fn hunt_trade_ship(game: &mut Game, ship_id: u32, target: u32, stepper: &mut PathStepper) {
    for _ in 0..2 {
        let Some(ship) = game.units().iter().find(|u| u.id == ship_id).cloned() else {
            return;
        };
        let Some(tgt) = game.units().iter().find(|u| u.id == target && u.active).cloned() else {
            return;
        };
        let dist = game.map.manhattan(ship.tile, tgt.tile);
        if dist <= 5 {
            game.capture_unit(&ship.owner_id, target);
            if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == ship_id) {
                u.record_trade_capture();
                u.target_unit = None;
            }
            return;
        }
        if dist <= 20 {
            if let Some(next) = best_water_neighbor_toward(game, ship.tile, tgt.tile) {
                game.move_unit(ship_id, next);
                continue;
            }
        }
        let (st, node) = game.water_step_with(stepper, ship.tile, tgt.tile);
        match st {
            PathStatus::Complete => {
                game.capture_unit(&ship.owner_id, target);
                if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == ship_id) {
                    u.record_trade_capture();
                    u.target_unit = None;
                }
                return;
            }
            PathStatus::Next => {
                if let Some(n) = node {
                    game.move_unit(ship_id, n);
                }
            }
            PathStatus::NotFound => {}
        }
    }
}

fn best_water_neighbor_toward(game: &Game, from: TileRef, to: TileRef) -> Option<TileRef> {
    let mut best = None;
    let mut best_dist = game.map.manhattan(from, to);
    let mut nbuf = [0u32; 4];
    let n = game.map.neighbors4(from, &mut nbuf);
    for i in 0..n {
        let nb = nbuf[i];
        if !game.map.is_water(nb) {
            continue;
        }
        let d = game.map.manhattan(nb, to);
        if d < best_dist {
            best_dist = d;
            best = Some(nb);
        }
    }
    best
}

fn air_stepper_next(
    game: &Game,
    from: TileRef,
    to: TileRef,
    seed: i32,
    path: &mut Option<Vec<TileRef>>,
    index: &mut usize,
    last_to: &mut Option<TileRef>,
) -> (PathStatus, Option<TileRef>) {
    if from == to {
        return (PathStatus::Complete, Some(to));
    }
    if *last_to != Some(to) {
        *path = None;
        *index = 0;
        *last_to = Some(to);
    }
    if path.is_none() {
        *path = Some(air_find_path(game, from, to, seed));
        *index = if path.as_ref().unwrap().first() == Some(&from) {
            1
        } else {
            0
        };
    }
    let p = path.as_ref().unwrap();
    if *index > 0 && p.get(*index - 1) != Some(&from) {
        *path = None;
        *index = 0;
        *last_to = Some(to);
        return air_stepper_next(game, from, to, seed, path, index, last_to);
    }
    if *index >= p.len() {
        return (PathStatus::Complete, Some(to));
    }
    let node = p[*index];
    *index += 1;
    (PathStatus::Next, Some(node))
}

fn air_find_path(game: &Game, from: TileRef, to: TileRef, seed: i32) -> Vec<TileRef> {
    let mut rng = PseudoRandom::new(seed);
    let mut path = vec![from];
    let mut current = from;
    while current != to {
        let next = air_compute_next(game, current, to, &mut rng);
        if next == current {
            break;
        }
        current = next;
        path.push(current);
    }
    path
}

fn air_compute_next(game: &Game, from: TileRef, to: TileRef, rng: &mut PseudoRandom) -> TileRef {
    let x = game.map.x(from) as i32;
    let y = game.map.y(from) as i32;
    let dst_x = game.map.x(to) as i32;
    let dst_y = game.map.y(to) as i32;
    if x == dst_x && y == dst_y {
        return to;
    }
    let mut next_x = x;
    let mut next_y = y;
    let ratio = 1 + (dst_y - y).abs() / ((dst_x - x).abs() + 1);
    if x == dst_x {
        next_y += if y < dst_y { 1 } else { -1 };
    } else if y == dst_y {
        next_x += if x < dst_x { 1 } else { -1 };
    } else if rng.chance(ratio) {
        next_x += if x < dst_x { 1 } else { -1 };
    } else {
        next_y += if y < dst_y { 1 } else { -1 };
    }
    game.map
        .ref_xy(next_x as u32, next_y as u32)
        .unwrap_or(from)
}

fn apply_shell_damage(
    game: &mut Game,
    target: u32,
    seed: i32,
    shooter_owner: &str,
    shooter: u32,
) {
    let Some(t) = game.units().iter().find(|u| u.id == target).cloned() else {
        return;
    };
    if !t.active {
        return;
    }
    let shooter_vet = game
        .units()
        .iter()
        .find(|u| u.id == shooter && u.active && u.ty == UnitType::Warship)
        .map(|u| u.veterancy)
        .unwrap_or(0);
    let mut rng = PseudoRandom::new(seed);
    let roll = rng.next_int(1.0, 6.0);
    let mut damage = (roll - 1) * 25 + 200;
    if shooter_vet > 0 {
        damage = (damage * (100 + shooter_vet as i32 * WARSHIP_VETERANCY_SHELL_DAMAGE_BONUS)) / 100;
    }
    let target_ty = t.ty;
    if unit_has_health(t.ty) {
        let mut killed = false;
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == target) {
            u.health -= damage;
            if u.health <= 0 {
                killed = true;
            }
        }
        if killed {
            game.delete_unit_by(target, Some(shooter_owner));
            if let Some(s) = game
                .units_mut()
                .iter_mut()
                .find(|u| u.id == shooter && u.active && u.ty == UnitType::Warship)
            {
                s.record_kill(target_ty);
            }
        }
    } else {
        game.delete_unit_by(target, Some(shooter_owner));
        if let Some(s) = game
            .units_mut()
            .iter_mut()
            .find(|u| u.id == shooter && u.active && u.ty == UnitType::Warship)
        {
            s.record_kill(target_ty);
        }
    }
}

fn patrol_warship(
    game: &mut Game,
    id: u32,
    patrol: TileRef,
    rng: &mut PseudoRandom,
    stepper: &mut PathStepper,
) {
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return;
    };
    if ship.target_tile.is_none() {
        let picked = random_warship_tile(game, ship.tile, patrol, rng, false);
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
            u.target_tile = picked;
            u.warship_state = WarshipMode::Patrolling;
        }
    }
    let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
        return;
    };
    let Some(dest) = ship.target_tile else {
        return;
    };
    let (st, node) = game.water_step_with(stepper, ship.tile, dest);
    match st {
        PathStatus::Complete => {
            if let Some(n) = node {
                game.move_unit(id, n);
            }
            if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                u.target_tile = None;
            }
        }
        PathStatus::Next => {
            if let Some(n) = node {
                game.move_unit(id, n);
            }
        }
        PathStatus::NotFound => {
            if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                u.target_tile = None;
            }
        }
    }
}

fn random_warship_tile(
    game: &Game,
    from: TileRef,
    patrol: TileRef,
    rng: &mut PseudoRandom,
    allow_shoreline: bool,
) -> Option<TileRef> {
    let mut range = WARSHIP_PATROL_RANGE as i32;
    let mut attempts = 0i32;
    let mut expand = 0i32;
    let ship_c = crate::spatial::get_water_component(
        &game.map,
        &game.mini_map,
        &game.mini_water,
        from,
    );
    while expand < 3 {
        let x = game.map.x(patrol) as i32 + rng.next_int((-range / 2) as f64, (range / 2) as f64);
        let y = game.map.y(patrol) as i32 + rng.next_int((-range / 2) as f64, (range / 2) as f64);
        if x < 0 || y < 0 {
            continue;
        }
        let Ok(tile) = game.map.ref_xy(x as u32, y as u32) else {
            continue;
        };
        if !game.map.is_water(tile) || (!allow_shoreline && game.map.is_shoreline(tile)) {
            attempts += 1;
            if attempts == 500 {
                expand += 1;
                attempts = 0;
                range += range / 2;
            }
            continue;
        }
        if let Some(wc) = ship_c {
            if !crate::spatial::has_water_component(
                &game.map,
                &game.mini_map,
                &game.mini_water,
                tile,
                wc,
            ) {
                attempts += 1;
                if attempts == 500 {
                    expand += 1;
                    attempts = 0;
                    range += range / 2;
                }
                continue;
            }
        }
        return Some(tile);
    }
    if !allow_shoreline {
        return random_warship_tile(game, from, patrol, rng, true);
    }
    None
}

fn maybe_break_nuke_alliances(game: &mut Game, owner: &str, dst: TileRef, ty: UnitType) {
    game.apply_nuke_blast_relations(owner, dst, ty);
}

fn nuke_death_factor(ty: UnitType, humans: f64, tiles_owned: u32, max_troops: f64) -> f64 {
    if ty != UnitType::MirvWarhead {
        return (5.0 * humans) / (tiles_owned as f64).max(1.0);
    }
    let target = 0.03 * max_troops;
    let excess = (humans - target).max(0.0);
    let normalized = if max_troops > 0.0 {
        excess / max_troops
    } else {
        0.0
    };
    500.0 * (1.0 - js_exp(-2.0 * normalized))
}

fn nuke_tiles_to_destroy(game: &mut Game, dst: TileRef, ty: UnitType) -> Vec<TileRef> {
    let (inner, outer) = nuke_magnitudes(ty);
    let inner2 = inner as i64 * inner as i64;
    let outer2 = outer as i64 * outer as i64;
    let mut rng = PseudoRandom::new(game.ticks() as i32);
    if game.later.water_nukes {
        const NUM_SAMPLES: usize = 16;
        let mut radii_sq = [0.0f64; NUM_SAMPLES];
        for r in radii_sq.iter_mut() {
            *r = rng.next_float(inner2 as f64, outer2 as f64);
        }
        let prev = radii_sq;
        for i in 0..NUM_SAMPLES {
            let l = (i + NUM_SAMPLES - 1) % NUM_SAMPLES;
            let r = (i + 1) % NUM_SAMPLES;
            radii_sq[i] = prev[i] * 0.6 + prev[l] * 0.2 + prev[r] * 0.2;
        }
        let cx = game.map.x(dst) as i32;
        let cy = game.map.y(dst) as i32;
        let outer_i = outer as i32;
        let x0 = (cx - outer_i).max(0);
        let y0 = (cy - outer_i).max(0);
        let x1 = (cx + outer_i).min(game.map.width() as i32 - 1);
        let y1 = (cy + outer_i).min(game.map.height() as i32 - 1);
        let mut result = Vec::new();
        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px - cx;
                let dy = py - cy;
                let d2 = dx as i64 * dx as i64 + dy as i64 * dy as i64;
                if d2 > outer2 {
                    continue;
                }
                if d2 > inner2 {
                    let angle = (dy as f64).atan2(dx as f64) + std::f64::consts::PI;
                    let t = (angle / (2.0 * std::f64::consts::PI)) * NUM_SAMPLES as f64;
                    let i0 = (t.floor() as usize) % NUM_SAMPLES;
                    let i1 = (i0 + 1) % NUM_SAMPLES;
                    let frac = t - t.floor();
                    let threshold = radii_sq[i0] * (1.0 - frac) + radii_sq[i1] * frac;
                    if d2 as f64 > threshold {
                        continue;
                    }
                }
                if let Ok(tile) = game.map.ref_xy(px as u32, py as u32) {
                    if !game.map.is_impassable(tile) {
                        result.push(tile);
                    }
                }
            }
        }
        return result;
    }
    let mut seen = vec![false; game.map.len()];
    let mut order = Vec::new();
    let mut stack = Vec::new();
    let consider = |n: TileRef, rng: &mut PseudoRandom, game: &Game| -> bool {
        let dx = game.map.x(n) as i64 - game.map.x(dst) as i64;
        let dy = game.map.y(n) as i64 - game.map.y(dst) as i64;
        let d2 = dx * dx + dy * dy;
        d2 <= outer2 && (d2 <= inner2 || rng.chance(2)) && !game.map.is_impassable(n)
    };
    if consider(dst, &mut rng, game) {
        seen[dst as usize] = true;
        order.push(dst);
        stack.push(dst);
    }
    let mut nbuf = [0u32; 4];
    while let Some(curr) = stack.pop() {
        let n = game.map.neighbors4(curr, &mut nbuf);
        for i in 0..n {
            let nb = nbuf[i];
            if !seen[nb as usize] && consider(nb, &mut rng, game) {
                seen[nb as usize] = true;
                order.push(nb);
                stack.push(nb);
            }
        }
    }
    order
}

fn apply_nuke_deaths(game: &mut Game, ty: UnitType, impacted: &[(String, u32)]) {
    for (pid, num) in impacted {
        let num = *num;
        if num == 0 {
            continue;
        }
        let tiles_now = game.player(pid).map(|p| p.num_tiles()).unwrap_or(0);
        let tiles_before = tiles_now + num;
        let max_troops = game.player_max_troops(pid);
        let transports: Vec<u32> = game
            .units()
            .iter()
            .filter(|u| u.active && u.ty == UnitType::TransportShip && u.owner_id == *pid)
            .map(|u| u.id)
            .collect();
        let mut ship_troops: HashMap<u32, f64> = HashMap::new();
        for &id in &transports {
            if let Some(u) = game.units().iter().find(|u| u.id == id) {
                ship_troops.insert(id, u.troops);
            }
        }
        for i in 0..num {
            let left = tiles_before - i;
            let humans = game.player(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
            let deaths = nuke_death_factor(ty, humans, left, max_troops);
            if let Some(p) = game.player_mut(pid) {
                p.remove_troops(deaths.floor() as i64);
                for atk in p.outgoing.iter_mut() {
                    let d = nuke_death_factor(ty, atk.troops, left, max_troops);
                    atk.troops = (atk.troops - d).max(0.0);
                }
            }
            for id in &transports {
                if let Some(t) = ship_troops.get_mut(id) {
                    let d = nuke_death_factor(ty, *t, left, max_troops);
                    *t = (*t - d).max(0.0);
                }
            }
        }
        for (id, troops) in ship_troops {
            if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
                u.troops = troops;
            }
        }
    }
}

fn detonate(game: &mut Game, nuke_id: u32, ty: UnitType, dst: TileRef) {
    let (_inner, outer) = nuke_magnitudes(ty);
    let outer2 = outer as i64 * outer as i64;
    let to_destroy = nuke_tiles_to_destroy(game, dst, ty);
    let mut impacted: Vec<(String, u32)> = Vec::new();
    for t in to_destroy {
        let owner = game.map.owner_id(t);
        if owner != 0 {
            if let Some(pid) = game.player_by_small(owner).map(|p| p.id().to_string()) {
                if let Some((_, n)) = impacted.iter_mut().find(|(id, _)| *id == pid) {
                    *n += 1;
                } else {
                    impacted.push((pid.clone(), 1));
                }
                game.relinquish_tile(&pid, t);
            }
        }
        if game.map.is_land(t) {
            game.set_fallout(t, true);
        }
    }
    apply_nuke_deaths(game, ty, &impacted);
    let mut killed = Vec::new();
    for u in game.units() {
        if !u.active || u.id == nuke_id {
            continue;
        }
        match u.ty {
            UnitType::AtomBomb
            | UnitType::HydrogenBomb
            | UnitType::MirvWarhead
            | UnitType::Mirv
            | UnitType::SamMissile => continue,
            _ => {}
        }
        let dx = game.map.x(dst) as i64 - game.map.x(u.tile) as i64;
        let dy = game.map.y(dst) as i64 - game.map.y(u.tile) as i64;
        if dx * dx + dy * dy < outer2 {
            killed.push(u.id);
        }
    }
    for id in killed {
        game.delete_unit(id);
    }
    game.delete_unit(nuke_id);
}

const MIRV_WARHEAD_RANGE: i32 = 1500;
const MIRV_WARHEAD_RANGE2: i64 = 1500 * 1500;
const MIRV_WARHEAD_MIN_SPREAD: i32 = 55;
const MIRV_WARHEAD_COUNT: usize = 350;

fn js_round(x: f64) -> i32 {
    (x + 0.5).floor() as i32
}

fn mirv_target_overlapping(game: &Game, x: i32, y: i32, taken: &[TileRef]) -> bool {
    for &existing in taken {
        let ex = game.map.x(existing) as i32;
        let ey = game.map.y(existing) as i32;
        if (x - ex).abs() + (y - ey).abs() < MIRV_WARHEAD_MIN_SPREAD {
            return true;
        }
    }
    false
}

fn try_mirv_warhead_target(
    game: &Game,
    rng: &mut PseudoRandom,
    base_x: i32,
    base_y: i32,
    target_sid: u16,
    taken: &[TileRef],
) -> Option<TileRef> {
    for _ in 0..100 {
        let r1 = rng.next();
        let r2 = (r1 * 15_485_863.0) % 1.0;
        let range = MIRV_WARHEAD_RANGE as f64;
        let x = js_round(r1 * range * 2.0 - range + base_x as f64);
        let y = js_round(r2 * range * 2.0 - range + base_y as f64);
        if x < 0 || y < 0 {
            continue;
        }
        let Ok(tile) = game.map.ref_xy(x as u32, y as u32) else {
            continue;
        };
        if !game.map.is_land(tile) {
            continue;
        }
        let dx = (x - base_x) as i64;
        let dy = (y - base_y) as i64;
        if dx * dx + dy * dy > MIRV_WARHEAD_RANGE2 {
            continue;
        }
        if game.map.owner_id(tile) != target_sid {
            continue;
        }
        if mirv_target_overlapping(game, x, y, taken) {
            continue;
        }
        return Some(tile);
    }
    None
}

/// Official `MirvExecution.selectDestinations` + `separate`.
fn drop_mirv_warheads(
    game: &mut Game,
    owner: &str,
    dst: TileRef,
    src: TileRef,
    rng: Option<&mut PseudoRandom>,
) {
    let Some(rng) = rng else { return };
    let target_sid = game.map.owner_id(dst);
    let base_x = game.map.x(dst) as i32;
    let base_y = game.map.y(dst) as i32;
    let mut targets: Vec<TileRef> = vec![dst];
    for _ in 0..1000 {
        if targets.len() >= MIRV_WARHEAD_COUNT {
            break;
        }
        if let Some(tile) =
            try_mirv_warhead_target(game, rng, base_x, base_y, target_sid, &targets)
        {
            targets.push(tile);
        }
    }
    targets.sort_by_key(|t| std::cmp::Reverse(game.map.manhattan(*t, dst)));
    let base_speed = nuke_speed(UnitType::MirvWarhead);
    for (i, tile) in targets.into_iter().enumerate() {
        let speed = base_speed + ((i as f64 / MIRV_WARHEAD_COUNT as f64) * 5.0).floor() as u32;
        let wait = rng.next_int(0.0, 15.0) as u32;
        game.enqueue_later(LaterExec::Nuke {
            active: true,
            owner: owner.to_string(),
            ty: UnitType::MirvWarhead,
            dst: tile,
            src: Some(src),
            unit_id: None,
            speed,
            wait,
            up: true,
            path: None,
            inited: false,
        });
    }
}

fn sam_missile_blocks_detonate(
    game: &Game,
    nuke_id: u32,
    dst: TileRef,
    nuke_owner: &str,
) -> bool {
    let r2 = (DEFAULT_SAM_MISSILE_SPEED as i64) * (DEFAULT_SAM_MISSILE_SPEED as i64);
    game.units().iter().any(|u| {
        if !u.active || u.ty != UnitType::SamMissile || u.owner_id == nuke_owner {
            return false;
        }
        if u.target_unit != Some(nuke_id) {
            return false;
        }
        let dx = game.map.x(dst) as i64 - game.map.x(u.tile) as i64;
        let dy = game.map.y(dst) as i64 - game.map.y(u.tile) as i64;
        dx * dx + dy * dy <= r2
    })
}

fn nuke_tile_targetable(game: &Game, src: Option<TileRef>, dst: TileRef, tile: TileRef) -> bool {
    let r2 = (DEFAULT_NUKE_TARGETABLE_RANGE as i64) * (DEFAULT_NUKE_TARGETABLE_RANGE as i64);
    let dx = game.map.x(tile) as i64 - game.map.x(dst) as i64;
    let dy = game.map.y(tile) as i64 - game.map.y(dst) as i64;
    if dx * dx + dy * dy < r2 {
        return true;
    }
    if let Some(s) = src {
        let dx = game.map.x(s) as i64 - game.map.x(tile) as i64;
        let dy = game.map.y(s) as i64 - game.map.y(tile) as i64;
        return dx * dx + dy * dy < r2;
    }
    false
}

fn sam_compute_intercept(
    game: &Game,
    sam_tile: TileRef,
    nuke: &crate::units::Unit,
    range2: f64,
) -> Option<(u32, TileRef)> {
    let traj = &nuke.trajectory;
    if traj.is_empty() {
        return None;
    }
    let current = nuke.trajectory_index.min(traj.len().saturating_sub(1));
    let max_i = traj.len().saturating_sub(2);
    let src = traj.first().copied();
    let dst = nuke.target_tile.unwrap_or(*traj.last().unwrap());
    for i in current..=max_i {
        let tile = traj[i];
        if !nuke_tile_targetable(game, src, dst, tile) {
            continue;
        }
        let dx = game.map.x(sam_tile) as i64 - game.map.x(tile) as i64;
        let dy = game.map.y(sam_tile) as i64 - game.map.y(tile) as i64;
        if (dx * dx + dy * dy) as f64 > range2 {
            continue;
        }
        let nuke_ticks = (i - current) as i32;
        let sam_ticks = ((game.map.manhattan(sam_tile, tile) as f64)
            / DEFAULT_SAM_MISSILE_SPEED as f64)
            .ceil() as i32;
        let wait = nuke_ticks - sam_ticks;
        if wait >= 0 {
            return Some((wait as u32, tile));
        }
    }
    let last = traj[traj.len() - 1];
    if nuke_tile_targetable(game, src, dst, last) {
        let dx = game.map.x(sam_tile) as i64 - game.map.x(last) as i64;
        let dy = game.map.y(sam_tile) as i64 - game.map.y(last) as i64;
        if (dx * dx + dy * dy) as f64 <= range2 && max_i >= current {
            let inflight = traj[max_i];
            if nuke_tile_targetable(game, src, dst, inflight) {
                let nuke_ticks = (max_i - current) as i32;
                let sam_ticks = ((game.map.manhattan(sam_tile, inflight) as f64)
                    / DEFAULT_SAM_MISSILE_SPEED as f64)
                    .ceil() as i32;
                let wait = nuke_ticks - sam_ticks;
                if wait >= 0 {
                    return Some((wait as u32, inflight));
                }
            }
        }
    }
    None
}

fn sam_valid_targets(game: &mut Game, sam: &crate::units::Unit, ticks: u32) -> Vec<(u32, TileRef)> {
    let range = sam_range(sam.level);
    let range2 = range * range;
    let detect = (MAX_SAM_RANGE as u32) * 4;
    let nearby = game.later_grid_nearby(
        sam.tile,
        detect,
        &[
            UnitType::AtomBomb,
            UnitType::HydrogenBomb,
            UnitType::MirvWarhead,
        ],
    );
    let nearby_ids: HashSet<u32> = nearby.iter().map(|(id, _)| *id).collect();
    if let Some(cache) = game.later.sam_intercept.get_mut(&sam.id) {
        cache.retain(|nid, _| nearby_ids.contains(nid));
    }
    let mut scored: Vec<(f64, u32, TileRef)> = Vec::new();
    for (nid, _) in nearby {
        let Some(nuke) = game.units().iter().find(|u| u.id == nid).cloned() else {
            continue;
        };
        if nuke.targeted_by_sam || nuke.owner_id == sam.owner_id {
            continue;
        }
        if game.is_allied(&sam.owner_id, &nuke.owner_id) {
            continue;
        }
        let cached = game
            .later
            .sam_intercept
            .get(&sam.id)
            .and_then(|m| m.get(&nid))
            .cloned();
        if let Some(c) = cached {
            match c {
                None => continue,
                Some((due, tile)) => {
                    if due == ticks || due == ticks + 1 {
                        scored.push((sam_target_score(game, sam, &nuke), nid, tile));
                        if let Some(m) = game.later.sam_intercept.get_mut(&sam.id) {
                            m.remove(&nid);
                        }
                        continue;
                    }
                    if due > ticks {
                        continue;
                    }
                    if let Some(m) = game.later.sam_intercept.get_mut(&sam.id) {
                        m.remove(&nid);
                    }
                }
            }
        }
        if let Some((wait, tile)) = sam_compute_intercept(game, sam.tile, &nuke, range2) {
            if wait <= 1 {
                scored.push((sam_target_score(game, sam, &nuke), nid, tile));
            } else {
                game.later
                    .sam_intercept
                    .entry(sam.id)
                    .or_default()
                    .insert(nid, Some((wait + ticks, tile)));
            }
        } else {
            game.later
                .sam_intercept
                .entry(sam.id)
                .or_default()
                .insert(nid, None);
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, id, t)| (id, t)).collect()
}

fn sam_target_score(game: &Game, sam: &crate::units::Unit, nuke: &crate::units::Unit) -> f64 {
    let dest = nuke
        .target_tile
        .or_else(|| nuke.trajectory.last().copied())
        .unwrap_or(sam.tile);
    let dist = game.map.manhattan(sam.tile, dest);
    let type_bonus = if nuke.ty == UnitType::HydrogenBomb {
        70_001.0
    } else {
        0.0
    };
    let distance_bonus = (200_000.0 - dist as f64 * 1000.0).max(0.0);
    let time_to_explode = nuke
        .trajectory
        .len()
        .saturating_sub(nuke.trajectory_index)
        .max(1);
    let urgency = (10_000.0 - time_to_explode as f64 * 100.0).max(0.0);
    type_bonus + distance_bonus + urgency
}
