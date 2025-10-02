from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import Config

Pos = Tuple[int, int]
DIRS_8 = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
STAY = (0, 0)

PREY_STATIC = 0  # +1 pt
PREY_ERRATIC = 1  # +2 pts
PREY_FLEE = 2  # +3 pts

PREY_POINTS = {
    PREY_STATIC: 1,
    PREY_ERRATIC: 2,
    PREY_FLEE: 3,
}


def in_bounds(p: Pos, n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


@dataclass
class Prey:
    pos: Pos
    ptype: int
    ttl: int


class PreyLogic2:
    """Logique avancée des proies avec TTL et fuyantes."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.preys: List[Prey] = []

        self._spawn_cycle = [PREY_STATIC, PREY_ERRATIC, PREY_FLEE]
        self._spawn_idx = 0
        self._spawn_interval: float = float(cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick: int = int(self._spawn_interval)

        self._ttl_static = int(cfg.TTL_STATIC)
        self._ttl_erratic = int(cfg.TTL_ERRATIC)
        self._ttl_flee = int(cfg.TTL_FLEE)

        self._decay_static = float(cfg.HP_DECAY_STATIC)
        self._decay_moving = float(cfg.HP_DECAY_MOVING)

    # --- cycle de vie ---
    def reset(self, env) -> None:
        self.preys.clear()
        self._spawn_idx = 0

        occ = {env.wolf}
        for _ in range(self.cfg.N_PREYS_START):
            p = self._sample_free_cell(env, occ)
            if p is None:
                break
            ptype = self._spawn_cycle[self._spawn_idx]
            self._spawn_idx = (self._spawn_idx + 1) % len(self._spawn_cycle)
            ttl = self._initial_ttl_for(ptype)
            self.preys.append(Prey(p, ptype, ttl))
            occ.add(p)

        self._spawn_interval = float(self.cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick = int(self._spawn_interval)

    def step_begin(self, env) -> None:
        self._decrement_and_purge_ttl()

    def step_end(self, env) -> None:
        return None

    # --- caractéristiques ---
    def hp_decay(self, env) -> float:
        moved = env.wolf != env._wolf_prev
        return self._decay_moving if moved else self._decay_static

    @property
    def next_spawn_tick(self) -> int:
        return self._next_spawn_tick

    # --- interactions ---
    def nearest_prey(self, env, pos: Optional[Pos] = None) -> tuple[int, Optional[Pos]]:
        if pos is None:
            pos = env.wolf
        if not self.preys:
            return 10**9, None
        best_d = 10**9
        best_p: Optional[Pos] = None
        for pr in self.preys:
            d = cheb(pos, pr.pos)
            if d < best_d:
                best_d, best_p = d, pr.pos
        return best_d, best_p

    def eat_if_in_reach(self, env) -> None:
        if env._ate_this_tick:
            return
        for idx, prey in enumerate(self.preys):
            if cheb(env.wolf, prey.pos) <= 1:
                self.preys.pop(idx)
                self._apply_heal(env)
                env.score += PREY_POINTS.get(prey.ptype, 1)
                env._ate_this_tick = True
                break

    def move_preys(self, env) -> None:
        wolf_cell = env.wolf
        n = env.n

        occupied = {pr.pos for pr in self.preys}
        order = list(range(len(self.preys)))
        self.rng.shuffle(order)

        for idx in order:
            pr = self.preys[idx]
            start = pr.pos

            if pr.ptype == PREY_STATIC:
                continue

            if pr.ptype == PREY_ERRATIC:
                if self.cfg.PREY_MOVES_EVERY_OTHER_TICK and (env.tick % 2 == 1):
                    continue
            else:
                if (env.tick % self.cfg.FLEE_MOVE_PERIOD) == self.cfg.FLEE_MOVE_REST_PHASE:
                    continue

            if self.rng.random() >= self.cfg.PREY_MOVE_PROB:
                continue

            occupied.discard(start)

            if pr.ptype == PREY_ERRATIC:
                new_pos = self._step_erratic(start, wolf_cell, occupied, n)
            else:
                if cheb(start, wolf_cell) <= self.cfg.FLEE_RADIUS:
                    new_pos = self._step_flee(start, wolf_cell, occupied, n)
                else:
                    new_pos = self._step_erratic(start, wolf_cell, occupied, n)

            if new_pos is None:
                if start != wolf_cell and start not in occupied:
                    pr.pos = start
                    occupied.add(start)
                else:
                    candidates = [
                        (start[0] + dx, start[1] + dy) for dx, dy in ([STAY] + DIRS_8)
                    ]
                    candidates = [
                        c for c in candidates if in_bounds(c, n) and c != wolf_cell and c not in occupied
                    ]
                    if candidates:
                        choice = candidates[self.rng.integers(0, len(candidates))]
                        pr.pos = choice
                        occupied.add(choice)
                    else:
                        pr.pos = start
                        occupied.add(start)
            else:
                pr.pos = new_pos
                occupied.add(new_pos)

    def maybe_spawn(self, env) -> None:
        if len(self.preys) >= self.cfg.MAX_PREYS:
            return
        if env.tick >= self._next_spawn_tick:
            occ = {env.wolf, *(pr.pos for pr in self.preys)}
            p = self._sample_free_cell(env, occ)
            if p is not None:
                ptype = self._spawn_cycle[self._spawn_idx]
                self._spawn_idx = (self._spawn_idx + 1) % len(self._spawn_cycle)
                ttl = self._initial_ttl_for(ptype)
                self.preys.append(Prey(p, ptype, ttl))
            self._spawn_interval = max(1.0, self._spawn_interval * self.cfg.SPAWN_INTERVAL_GROWTH)
            self._next_spawn_tick = env.tick + int(round(self._spawn_interval))

    # --- utilitaires ---
    def _apply_heal(self, env) -> None:
        missing = float(self.cfg.HP_MAX) - env.hp
        if missing <= 0:
            return
        heal = missing * float(self.cfg.EAT_HEAL_MISSING_FRACTION)
        if heal <= 0:
            return
        env.hp = min(float(self.cfg.HP_MAX), env.hp + heal)

    def _sample_free_cell(self, env, occ: set) -> Optional[Pos]:
        tries = 0
        while tries < 400:
            tries += 1
            p = (self.rng.integers(0, env.n), self.rng.integers(0, env.n))
            if p not in occ:
                return p
        return None

    def _initial_ttl_for(self, ptype: int) -> int:
        if ptype == PREY_STATIC:
            return self._ttl_static
        if ptype == PREY_ERRATIC:
            return self._ttl_erratic
        return self._ttl_flee

    def _decrement_and_purge_ttl(self) -> None:
        i = 0
        while i < len(self.preys):
            self.preys[i].ttl -= 1
            if self.preys[i].ttl <= 0:
                self.preys.pop(i)
            else:
                i += 1

    def _step_erratic(self, start: Pos, wolf: Pos, occupied: set, n: int) -> Optional[Pos]:
        candidates = [(start[0] + dx, start[1] + dy) for dx, dy in ([STAY] + DIRS_8)]
        candidates = [c for c in candidates if in_bounds(c, n) and c != wolf and c not in occupied]
        if not candidates:
            return None
        return candidates[self.rng.integers(0, len(candidates))]

    def _step_flee(self, start: Pos, wolf: Pos, occupied: set, n: int) -> Optional[Pos]:
        vx = sign(start[0] - wolf[0])
        vy = sign(start[1] - wolf[1])
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
