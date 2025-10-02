"""Prey logic for the classic survival environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import Config

Pos = Tuple[int, int]
DIRS_8 = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]
STAY = (0, 0)


def in_bounds(p: Pos, n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass
class BasicPrey:
    pos: Pos
    score_value: int = 1


class BasicPreyLogic:
    """Encapsulates spawn and movement logic for the basic environment."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.GRID_SIZE
        self.preys: List[BasicPrey] = []
        self._spawn_interval: float = float(cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick: int = int(self._spawn_interval)

    @property
    def next_spawn_tick(self) -> int:
        return self._next_spawn_tick

    def reset(self, wolf_pos: Pos) -> None:
        self.preys = []
        occ = {wolf_pos}
        while len(self.preys) < self.cfg.N_PREYS_START:
            p = (
                self.rng.integers(0, self.n),
                self.rng.integers(0, self.n),
            )
            if p not in occ:
                self.preys.append(BasicPrey(p))
                occ.add(p)

        self._spawn_interval = float(self.cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick = int(self._spawn_interval)

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

    def eat_if_in_reach(self, wolf_pos: Pos) -> Optional[BasicPrey]:
        for idx, prey in enumerate(self.preys):
            if cheb(wolf_pos, prey.pos) <= 1:
                return self.preys.pop(idx)
        return None

    def move(self, tick: int, wolf_pos: Pos) -> None:
        if self.cfg.PREY_MOVES_EVERY_OTHER_TICK and (tick % 2 == 1):
            return

        n = self.n
        wolf_cell = wolf_pos
        occupied = {prey.pos for prey in self.preys}
        order = list(range(len(self.preys)))
        self.rng.shuffle(order)

        for idx in order:
            prey = self.preys[idx]
            start = prey.pos
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
            prey.pos = choice
            occupied.add(choice)

    def spawn(self, tick: int, wolf_pos: Pos) -> None:
        if len(self.preys) >= self.cfg.MAX_PREYS:
            return
        if tick < self._next_spawn_tick:
            return

        occ = {wolf_pos, *(prey.pos for prey in self.preys)}
        tries = 0
        while tries < 200:
            tries += 1
            p = (
                self.rng.integers(0, self.n),
                self.rng.integers(0, self.n),
            )
            if p not in occ:
                self.preys.append(BasicPrey(p))
                break

        self._spawn_interval = max(1.0, self._spawn_interval * self.cfg.SPAWN_INTERVAL_GROWTH)
        self._next_spawn_tick = tick + int(round(self._spawn_interval))


__all__ = [
    "BasicPrey",
    "BasicPreyLogic",
    "DIRS_8",
    "Pos",
    "STAY",
    "cheb",
    "in_bounds",
]
