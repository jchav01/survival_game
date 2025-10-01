from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from config import Config

Pos = Tuple[int, int]
DIRS_8 = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
STAY = (0,0)


def in_bounds(p: Pos, n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))


@dataclass
class Prey:
    pos: Pos


class SurvivalEnv:
    """
    Environnement minimaliste, sans notion de reward/capture.
    - State: positions loup + proies, barre de vie, tick.
    - Step: la politique externe met à jour la position du loup via `set_wolf` avant `step()`.
      `step()` applique faim, manger si à portée, déplacement des proies (1 tick/2), spawn éventuel.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n = cfg.GRID_SIZE
        self.rng = np.random.default_rng(cfg.RNG_SEED)

        self.tick: int = 0
        self.wolf: Pos = (0,0)
        self.hp: float = float(cfg.HP_MAX)
        self.preys: List[Prey] = []
        self.preys_eaten: int = 0
        self._ate_this_tick: bool = False

        # spawn scheduler (délai croissant)
        self._spawn_interval: float = float(cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick: int = int(self._spawn_interval)

    # --- API contrôle externe du loup ---
    def set_wolf(self, pos: Pos):
        x = max(0, min(self.n-1, int(pos[0])))
        y = max(0, min(self.n-1, int(pos[1])))
        self.wolf = (x, y)

    # --- Reset ---
    def reset(self):
        self.tick = 0
        self.hp = float(self.cfg.HP_MAX)
        self.preys = []
        self.preys_eaten = 0
        self._ate_this_tick = False

        # place wolf
        self.wolf = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))

        # place initial preys
        occ = {self.wolf}
        while len(self.preys) < self.cfg.N_PREYS_START:
            p = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))
            if p not in occ:
                self.preys.append(Prey(p))
                occ.add(p)

        # reset spawn schedule
        self._spawn_interval = float(self.cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick = int(self._spawn_interval)

    # --- Query helpers ---
    def nearest_prey(self, pos: Optional[Pos] = None) -> tuple[int, Optional[Pos]]:
        if pos is None:
            pos = self.wolf
        if not self.preys:
            return (10**9, None)
        best_d = 10**9
        best_p: Optional[Pos] = None
        for pr in self.preys:
            d = cheb(pos, pr.pos)
            if d < best_d:
                best_d = d
                best_p = pr.pos
        return best_d, best_p

    # --- Eating ---
    def _apply_heal(self):
        missing = float(self.cfg.HP_MAX) - self.hp
        if missing <= 0:
            return
        heal = missing * float(self.cfg.EAT_HEAL_MISSING_FRACTION)
        if heal <= 0:
            return
        self.hp = min(float(self.cfg.HP_MAX), self.hp + heal)

    def _eat_if_in_reach(self):
        """Mange au plus une proie adjacente (distance de Chebyshev <= 1)."""
        if self._ate_this_tick:
            return
        for idx, prey in enumerate(self.preys):
            if cheb(self.wolf, prey.pos) <= 1:
                self.preys.pop(idx)
                self._apply_heal()
                self.preys_eaten += 1
                self._ate_this_tick = True
                break

    # --- Preys movement (1 tick sur 2) ---
    def _preys_step(self):
        if self.cfg.PREY_MOVES_EVERY_OTHER_TICK and (self.tick % 2 == 1):
            # bougent un tick sur deux (ici: ticks impairs). Change si tu préfères pairs.
            return
        n = self.n
        wolf_cell = self.wolf
        occupied = {pr.pos for pr in self.preys}
        order = list(range(len(self.preys)))
        self.rng.shuffle(order)

        for idx in order:
            pr = self.preys[idx]
            start = pr.pos
            if self.rng.random() >= self.cfg.PREY_MOVE_PROB:
                continue
            candidates = [(start[0]+dx, start[1]+dy) for dx,dy in ([STAY] + DIRS_8)]
            candidates = [c for c in candidates if in_bounds(c, n) and c != wolf_cell]

            occupied.discard(start)
            free = [c for c in candidates if c not in occupied]
            if not free:
                occupied.add(start)
                continue
            choice = free[self.rng.integers(0, len(free))]
            pr.pos = choice
            occupied.add(choice)

    # --- Spawn schedule with increasing delay ---
    def _maybe_spawn(self):
        if len(self.preys) >= self.cfg.MAX_PREYS:
            return
        if self.tick >= self._next_spawn_tick:
            # spawn une proie sur une case libre
            occ = {self.wolf, *(pr.pos for pr in self.preys)}
            tries = 0
            while tries < 200:
                tries += 1
                p = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))
                if p not in occ:
                    self.preys.append(Prey(p))
                    break
            # planifier prochain spawn (délai croissant)
            self._spawn_interval = max(1.0, self._spawn_interval * self.cfg.SPAWN_INTERVAL_GROWTH)
            self._next_spawn_tick = self.tick + int(round(self._spawn_interval))

    # --- Step ---
    def step(self):
        self.tick += 1
        self._ate_this_tick = False
        # Décroissance de la vie
        self.hp = max(0.0, self.hp - self.cfg.HP_DECAY_PER_TICK)

        # Manger AVANT le déplacement des proies
        self._eat_if_in_reach()

        # Mouvement des proies (un tick sur deux)
        self._preys_step()

        # Spawn éventuel (délai croissant)
        self._maybe_spawn()

        # Manger APRÈS déplacement (au cas où une proie se rapproche)
        self._eat_if_in_reach()

        # Terminaison quand hp tombe à 0
        done = (self.hp <= 0)
        return done
