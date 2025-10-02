from __future__ import annotations

from typing import Optional, Protocol, Tuple, Type, Union

import numpy as np

from config import Config
from preys1 import PreyLogic1
from preys2 import PreyLogic2

Pos = Tuple[int, int]


class PreyLogic(Protocol):
    preys: list

    def reset(self, env: "SurvivalEnv") -> None:
        ...

    def step_begin(self, env: "SurvivalEnv") -> None:
        ...

    def step_end(self, env: "SurvivalEnv") -> None:
        ...

    def hp_decay(self, env: "SurvivalEnv") -> float:
        ...

    @property
    def next_spawn_tick(self) -> int:
        ...

    def nearest_prey(self, env: "SurvivalEnv", pos: Optional[Pos] = None) -> tuple[int, Optional[Pos]]:
        ...

    def eat_if_in_reach(self, env: "SurvivalEnv") -> None:
        ...

    def move_preys(self, env: "SurvivalEnv") -> None:
        ...

    def maybe_spawn(self, env: "SurvivalEnv") -> None:
        ...


_PREY_LOGIC_MAP = {
    "preys1": PreyLogic1,
    "logic1": PreyLogic1,
    "1": PreyLogic1,
    "preys2": PreyLogic2,
    "logic2": PreyLogic2,
    "2": PreyLogic2,
}


class SurvivalEnv:
    """Environnement générique dont la logique des proies est interchangeable."""

    def __init__(self, cfg: Config, prey_logic: Union[str, PreyLogic, Type[PreyLogic], None] = None):
        self.cfg = cfg
        self.n = cfg.GRID_SIZE
        self.rng = np.random.default_rng(cfg.RNG_SEED)

        self.tick: int = 0
        self.wolf: Pos = (0, 0)
        self._wolf_prev: Pos = self.wolf
        self.hp: float = float(cfg.HP_MAX)
        self.score: int = 0
        self._ate_this_tick: bool = False

        self.prey_logic: PreyLogic = self._init_prey_logic(prey_logic)

    # --- Initialisation logique de proies ---
    def _init_prey_logic(self, selection: Union[str, PreyLogic, Type[PreyLogic], None]) -> PreyLogic:
        if isinstance(selection, (PreyLogic1, PreyLogic2)):
            return selection  # type: ignore[return-value]
        if selection is None:
            selection = getattr(self.cfg, "PREY_LOGIC", "preys1")
        if isinstance(selection, str):
            key = selection.lower()
            if key not in _PREY_LOGIC_MAP:
                raise ValueError(
                    f"Unknown prey logic '{selection}'. Available: {sorted(set(_PREY_LOGIC_MAP))}"
                )
            logic_cls = _PREY_LOGIC_MAP[key]
            return logic_cls(self.cfg, self.rng)  # type: ignore[return-value]
        if isinstance(selection, type):
            return selection(self.cfg, self.rng)  # type: ignore[return-value]
        return selection

    # --- Exposition minimale ---
    @property
    def preys(self):
        return self.prey_logic.preys

    @property
    def next_spawn_tick(self) -> int:
        return getattr(self.prey_logic, "next_spawn_tick", 0)

    # --- API contrôle externe du loup ---
    def set_wolf(self, pos: Pos):
        x = max(0, min(self.n - 1, int(pos[0])))
        y = max(0, min(self.n - 1, int(pos[1])))
        self.wolf = (x, y)

    # --- Reset ---
    def reset(self):
        self.tick = 0
        self.hp = float(self.cfg.HP_MAX)
        self.score = 0
        self._ate_this_tick = False

        self.wolf = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))
        self._wolf_prev = self.wolf

        self.prey_logic.reset(self)

    # --- Query helpers ---
    def nearest_prey(self, pos: Optional[Pos] = None) -> tuple[int, Optional[Pos]]:
        return self.prey_logic.nearest_prey(self, pos)

    # --- Step ---
    def step(self):
        self.tick += 1
        self._ate_this_tick = False

        self.prey_logic.step_begin(self)

        decay = self.prey_logic.hp_decay(self)
        self.hp = max(0.0, self.hp - decay)

        if self.hp > 0.0:
            self.prey_logic.eat_if_in_reach(self)

        self.prey_logic.move_preys(self)
        self.prey_logic.maybe_spawn(self)

        if self.hp > 0.0:
            self.prey_logic.eat_if_in_reach(self)

        self.prey_logic.step_end(self)

        done = self.hp <= 0.0
        self._wolf_prev = self.wolf
        return done
