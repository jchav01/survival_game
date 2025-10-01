from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from config import Config
from preys import AdvancedPreyLogic, Pos


class SurvivalEnvV2:
    """Advanced survival environment relying on :mod:`preys` for prey behaviour."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n = cfg.GRID_SIZE
        self.rng = np.random.default_rng(cfg.RNG_SEED)

        self.tick: int = 0
        self.wolf: Pos = (0, 0)
        self._wolf_prev: Pos = (0, 0)
        self.hp: float = float(cfg.HP_MAX)
        self.score: int = 0
        self._ate_this_tick: bool = False

        self.prey_logic = AdvancedPreyLogic(cfg, self.rng)
        self.preys = self.prey_logic.preys

        self._next_spawn_tick: int = self.prey_logic.next_spawn_tick

    def set_wolf(self, pos: Pos) -> None:
        x = max(0, min(self.n - 1, int(pos[0])))
        y = max(0, min(self.n - 1, int(pos[1])))
        self.wolf = (x, y)

    def reset(self) -> None:
        self.tick = 0
        self.hp = float(self.cfg.HP_MAX)
        self.score = 0
        self._ate_this_tick = False

        self.wolf = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))
        self._wolf_prev = self.wolf
        self.prey_logic.reset(self.wolf)
        self._next_spawn_tick = self.prey_logic.next_spawn_tick

    def nearest_prey(self, pos: Optional[Pos] = None) -> Tuple[int, Optional[Pos]]:
        return self.prey_logic.nearest_prey(self.wolf, pos)

    def _apply_heal(self) -> None:
        missing = float(self.cfg.HP_MAX) - self.hp
        if missing <= 0:
            return
        heal = missing * float(self.cfg.EAT_HEAL_MISSING_FRACTION)
        if heal <= 0:
            return
        self.hp = min(float(self.cfg.HP_MAX), self.hp + heal)

    def _eat_if_in_reach(self) -> None:
        if self._ate_this_tick:
            return
        prey = self.prey_logic.eat_if_in_reach(self.wolf)
        if prey is None:
            return
        self._apply_heal()
        self.score += prey.score_value
        self._ate_this_tick = True

    def step(self) -> bool:
        self.tick += 1
        self._ate_this_tick = False

        self.prey_logic.before_step()

        moved = self.wolf != self._wolf_prev
        decay = self.cfg.HP_DECAY_MOVING if moved else self.cfg.HP_DECAY_STATIC
        self.hp = max(0.0, self.hp - decay)
        self._wolf_prev = self.wolf

        if self.hp > 0.0:
            self._eat_if_in_reach()

        self.prey_logic.move(self.tick, self.wolf)
        self.prey_logic.spawn(self.tick, self.wolf)
        self._next_spawn_tick = self.prey_logic.next_spawn_tick

        if self.hp > 0.0:
            self._eat_if_in_reach()

        return self.hp <= 0.0


__all__ = ["SurvivalEnvV2"]
