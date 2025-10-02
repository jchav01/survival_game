"""Prey logic for the advanced survival environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import Config
from preys_logic1 import DIRS_8, STAY, cheb, in_bounds

Pos = Tuple[int, int]

PREY_STATIC = 0
PREY_ERRATIC = 1
PREY_FLEE = 2

PREY_POINTS = {
    PREY_STATIC: 1,
    PREY_ERRATIC: 2,
    PREY_FLEE: 3,
}


@dataclass
class AdvancedPrey:
    pos: Pos
    ptype: int
    ttl: int

    @property
    def score_value(self) -> int:
        return PREY_POINTS.get(self.ptype, 1)


class AdvancedPreyLogic:
    """Extended prey behaviour used by the second environment."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.GRID_SIZE
        self.preys: List[AdvancedPrey] = []

        self._spawn_cycle = [PREY_STATIC, PREY_ERRATIC, PREY_FLEE]
        self._spawn_idx = 0
        self._spawn_interval: float = float(cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick: int = int(self._spawn_interval)

        self._ttl_static = int(cfg.TTL_STATIC)
        self._ttl_erratic = int(cfg.TTL_ERRATIC)
        self._ttl_flee = int(cfg.TTL_FLEE)

    @property
    def next_spawn_tick(self) -> int:
        return self._next_spawn_tick

    def reset(self, wolf_pos: Pos) -> None:
        self.preys = []
        self._spawn_idx = 0

        occ = {wolf_pos}
        for _ in range(self.cfg.N_PREYS_START):
            p = self._sample_free_cell(occ)
            if p is None:
                break
            ptype = self._spawn_cycle[self._spawn_idx]
            self._spawn_idx = (self._spawn_idx + 1) % len(self._spawn_cycle)
            ttl = self._initial_ttl_for(ptype)
            self.preys.append(AdvancedPrey(p, ptype, ttl))
            occ.add(p)

        self._spawn_interval = float(self.cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick = int(self._spawn_interval)

    def _sample_free_cell(self, occupied: set[Pos]) -> Optional[Pos]:
        tries = 0
        while tries < 400:
            tries += 1
            p = (
                self.rng.integers(0, self.n),
                self.rng.integers(0, self.n),
            )
            if p not in occupied:
                return p
        return None

    def _initial_ttl_for(self, ptype: int) -> int:
        if ptype == PREY_STATIC:
            return self._ttl_static
        if ptype == PREY_ERRATIC:
            return self._ttl_erratic
        return self._ttl_flee

    def nearest_prey(self, wolf_pos: Pos, pos: Optional[Pos] = None) -> tuple[int, Optional[Pos]]:
        if pos is None:
            pos = wolf_pos
        if not self.preys:
            return 10**9, None
        best_d = 10**9
        best_p: Optional[Pos] = None
        for prey in self.preys:
            d = cheb(pos, prey.pos)
            if d < best_d:
                best_d = d
                best_p = prey.pos
        return best_d, best_p

    def before_step(self) -> None:
        i = 0
        while i < len(self.preys):
            self.preys[i].ttl -= 1
            if self.preys[i].ttl <= 0:
                self.preys.pop(i)
            else:
                i += 1

    def eat_if_in_reach(self, wolf_pos: Pos) -> Optional[AdvancedPrey]:
        for idx, prey in enumerate(self.preys):
            if cheb(wolf_pos, prey.pos) <= 1:
                return self.preys.pop(idx)
        return None

    def move(self, tick: int, wolf_pos: Pos) -> None:
        n = self.n
        wolf_cell = wolf_pos

        occupied = {prey.pos for prey in self.preys}
        order = list(range(len(self.preys)))
        self.rng.shuffle(order)

        for idx in order:
            prey = self.preys[idx]
            start = prey.pos

            if prey.ptype == PREY_STATIC:
                continue

            if prey.ptype == PREY_ERRATIC:
                if self.cfg.PREY_MOVES_EVERY_OTHER_TICK and (tick % 2 == 1):
                    continue
            else:
                if (tick % self.cfg.FLEE_MOVE_PERIOD) == self.cfg.FLEE_MOVE_REST_PHASE:
                    continue

            if self.rng.random() >= self.cfg.PREY_MOVE_PROB:
                continue

            occupied.discard(start)

            if prey.ptype == PREY_ERRATIC:
                new_pos = self._step_erratic(start, wolf_cell, occupied, n)
            else:
                if cheb(start, wolf_cell) <= self.cfg.FLEE_RADIUS:
                    new_pos = self._step_flee(start, wolf_cell, occupied, n)
                else:
                    new_pos = self._step_erratic(start, wolf_cell, occupied, n)

            if new_pos is None:
                if start != wolf_cell and start not in occupied:
                    prey.pos = start
                    occupied.add(start)
                else:
                    candidates = [
                        (start[0] + dx, start[1] + dy)
                        for dx, dy in ([STAY] + DIRS_8)
                        if in_bounds((start[0] + dx, start[1] + dy), n)
                        and (start[0] + dx, start[1] + dy) != wolf_cell
                        and (start[0] + dx, start[1] + dy) not in occupied
                    ]
                    if candidates:
                        choice = candidates[self.rng.integers(0, len(candidates))]
                        prey.pos = choice
                        occupied.add(choice)
                    else:
                        prey.pos = start
                        occupied.add(start)
            else:
                prey.pos = new_pos
                occupied.add(new_pos)

    def _step_erratic(
        self, start: Pos, wolf: Pos, occupied: set[Pos], n: int
    ) -> Optional[Pos]:
        candidates = [
            (start[0] + dx, start[1] + dy)
            for dx, dy in ([STAY] + DIRS_8)
            if in_bounds((start[0] + dx, start[1] + dy), n)
            and (start[0] + dx, start[1] + dy) != wolf
            and (start[0] + dx, start[1] + dy) not in occupied
        ]
        if not candidates:
            return None
        return candidates[self.rng.integers(0, len(candidates))]

    def _step_flee(
        self, start: Pos, wolf: Pos, occupied: set[Pos], n: int
    ) -> Optional[Pos]:
        vx = _sign(start[0] - wolf[0])
        vy = _sign(start[1] - wolf[1])
        if vx == 0 and vy == 0:
            return self._step_erratic(start, wolf, occupied, n)

        primary = (start[0] + vx, start[1] + vy)
        if in_bounds(primary, n) and primary not in occupied and primary != wolf:
            return primary

        candidates: List[Pos] = []
        if not in_bounds((start[0] + vx, start[1]), n):
            for vy2 in (-1, 1):
                alt = (start[0], start[1] + vy2)
                if in_bounds(alt, n) and alt not in occupied and alt != wolf:
                    candidates.append(alt)
        if not in_bounds((start[0], start[1] + vy), n):
            for vx2 in (-1, 1):
                alt = (start[0] + vx2, start[1])
                if in_bounds(alt, n) and alt not in occupied and alt != wolf:
                    candidates.append(alt)

        if candidates:
            return candidates[self.rng.integers(0, len(candidates))]

        better: List[Pos] = []
        same: List[Pos] = []
        d0 = cheb(start, wolf)
        for dx, dy in DIRS_8 + [STAY]:
            alt = (start[0] + dx, start[1] + dy)
            if not in_bounds(alt, n) or alt == wolf or alt in occupied:
                continue
            d = cheb(alt, wolf)
            if d > d0:
                better.append(alt)
            elif d == d0:
                same.append(alt)
        pool = better if better else same
        if pool:
            return pool[self.rng.integers(0, len(pool))]
        return None

    def spawn(self, tick: int, wolf_pos: Pos) -> None:
        if len(self.preys) >= self.cfg.MAX_PREYS:
            return
        if tick < self._next_spawn_tick:
            return

        occ = {wolf_pos, *(prey.pos for prey in self.preys)}
        p = self._sample_free_cell(occ)
        if p is not None:
            ptype = self._spawn_cycle[self._spawn_idx]
            self._spawn_idx = (self._spawn_idx + 1) % len(self._spawn_cycle)
            ttl = self._initial_ttl_for(ptype)
            self.preys.append(AdvancedPrey(p, ptype, ttl))

        self._spawn_interval = max(1.0, self._spawn_interval * self.cfg.SPAWN_INTERVAL_GROWTH)
        self._next_spawn_tick = tick + int(round(self._spawn_interval))


def _sign(x: int) -> int:
    if x == 0:
        return 0
    return 1 if x > 0 else -1


__all__ = [
    "AdvancedPrey",
    "AdvancedPreyLogic",
    "Pos",
    "PREY_ERRATIC",
    "PREY_FLEE",
    "PREY_POINTS",
    "PREY_STATIC",
]
