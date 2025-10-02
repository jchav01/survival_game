from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import Config

Pos = Tuple[int, int]
DIRS_8 = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
STAY = (0, 0)


def in_bounds(p: Pos, n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass
class Prey:
    pos: Pos


class PreyLogic1:
    """Logique historique des proies."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.preys: List[Prey] = []
        self._spawn_interval: float = float(cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick: int = int(self._spawn_interval)

    # --- cycle de vie ---
    def reset(self, env) -> None:
        self.preys.clear()
        occ = {env.wolf}
        while len(self.preys) < self.cfg.N_PREYS_START:
            p = (self.rng.integers(0, env.n), self.rng.integers(0, env.n))
            if p not in occ:
                self.preys.append(Prey(p))
                occ.add(p)

        self._spawn_interval = float(self.cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick = int(self._spawn_interval)

    def step_begin(self, env) -> None:
        # rien de spécifique pour cette logique
        return None

    def step_end(self, env) -> None:
        return None

    # --- caractéristiques ---
    def hp_decay(self, env) -> float:
        return float(self.cfg.HP_DECAY_PER_TICK)

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
                env.score += 1
                env._ate_this_tick = True
                break

    def move_preys(self, env) -> None:
        if self.cfg.PREY_MOVES_EVERY_OTHER_TICK and (env.tick % 2 == 1):
            return
        n = env.n
        wolf_cell = env.wolf
        occupied = {pr.pos for pr in self.preys}
        order = list(range(len(self.preys)))
        self.rng.shuffle(order)

        for idx in order:
            pr = self.preys[idx]
            start = pr.pos
            if self.rng.random() >= self.cfg.PREY_MOVE_PROB:
                continue
            candidates = [(start[0] + dx, start[1] + dy) for dx, dy in ([STAY] + DIRS_8)]
            candidates = [c for c in candidates if in_bounds(c, n) and c != wolf_cell]

            occupied.discard(start)
            free = [c for c in candidates if c not in occupied]
            if not free:
                occupied.add(start)
                continue
            choice = free[self.rng.integers(0, len(free))]
            pr.pos = choice
            occupied.add(choice)

    def maybe_spawn(self, env) -> None:
        if len(self.preys) >= self.cfg.MAX_PREYS:
            return
        if env.tick >= self._next_spawn_tick:
            occ = {env.wolf, *(pr.pos for pr in self.preys)}
            tries = 0
            while tries < 200:
                tries += 1
                p = (self.rng.integers(0, env.n), self.rng.integers(0, env.n))
                if p not in occ:
                    self.preys.append(Prey(p))
                    break
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
