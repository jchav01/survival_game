
# -*- coding: utf-8 -*-
"""
Gymnasium-style environment for Wolf-Prey on a toroidal grid.
Observation (vector A):
    obs = [hunger_norm, dx_norm, dy_norm, d_min_norm, prey_in_reach]
Action space (Discrete 9):
    0: stay
    1..8: sprint (2 cells) in 8 directions [N, NE, E, SE, S, SW, W, NW]
Episode ends when:
    - capture_quota reached
    - hunger >= HUNGER_MAX
    - step >= T_MAX
Proies move (random walk + simple flee), but NO spawn during an episode.
"""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple

# ---------------- Defaults (override via __init__) ----------------
DEFAULTS = dict(
    grid_size=60,
    n_preys=15,
    capture_quota=3,
    t_max=500,
    sprint_cost=0.35,
    sprint_hunger_bonus=3,
    eat_reward_max=5.0,
    hunger_norm=120.0,
    hunger_max=200,
    step_penalty=0.01,
    gamma=0.95,
    use_potential_shaping=True,
    potential_alpha=0.1,
    prey_move_prob=0.85,
    prey_flee_near=4,
    prey_flee_radius=8,
    prey_flee_eps=0.30,
    prey_flee_jitter_prob=0.15,
    seed=None,
)

DIRS_8 = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0,  1), (-1, 1), (-1, 0), (-1, -1)
]
STAY = (0, 0)

def wrap_xy(x: int, y: int, n: int) -> Tuple[int, int]:
    return (x % n, y % n)

def torus_delta(a: int, b: int, n: int) -> int:
    d = (b - a) % n
    return d - n if d > n//2 else d

def cheb_dist(a: Tuple[int,int], b: Tuple[int,int], n: int) -> int:
    dx = abs(torus_delta(a[0], b[0], n))
    dy = abs(torus_delta(a[1], b[1], n))
    return max(dx, dy)

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1.0 else x)

@dataclass
class Prey:
    pos: Tuple[int, int]

class WolfPreyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, **kwargs):
        cfg = DEFAULTS.copy()
        cfg.update(kwargs or {})
        self.cfg = cfg

        self.n = cfg["grid_size"]
        self.rng = np.random.default_rng(cfg["seed"])

        self.action_space = spaces.Discrete(9)  # 0 stay, 1..8 sprints
        # obs = [hunger_norm, dx_norm, dy_norm, d_min_norm, prey_in_reach]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # state
        self.wolf = (0, 0)
        self.hunger_ticks = 0
        self.preys: List[Prey] = []
        self.captures = 0
        self.steps = 0

        # render cache
        self._last_rgb = None

    # ------------- Helpers -------------
    def _random_free_positions(self, k: int) -> List[Tuple[int,int]]:
        seen = set()
        out: List[Tuple[int,int]] = []
        tries = 0
        while len(out) < k and tries < 5000:
            tries += 1
            p = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _nearest_prey(self, pos: Tuple[int,int]) -> Tuple[int, Tuple[int,int] | None]:
        if not self.preys:
            return 0, None
        best_d = 10**9
        best = None
        for pr in self.preys:
            d = cheb_dist(pos, pr.pos, self.n)
            if d < best_d:
                best_d, best = d, pr.pos
        return best_d, best

    def _sprint_dirs(self):
        return [(2*dx, 2*dy) for (dx,dy) in DIRS_8]

    def _decompose(self, mv: Tuple[int,int]) -> List[Tuple[int,int]]:
        dx, dy = mv
        steps = []
        length = max(abs(dx), abs(dy))
        if length == 0:
            return steps
        sx = 0 if dx == 0 else (1 if dx > 0 else -1)
        sy = 0 if dy == 0 else (1 if dy > 0 else -1)
        for _ in range(length):
            steps.append((sx, sy))
        return steps

    def _prey_in_reach(self, pos: Tuple[int,int]) -> bool:
        for pr in self.preys:
            if cheb_dist(pos, pr.pos, self.n) <= 1:
                return True
        return False

    def _eat_here(self):
        i = 0
        count = 0
        while i < len(self.preys):
            if cheb_dist(self.wolf, self.preys[i].pos, self.n) <= 1:
                self.preys.pop(i)
                self.hunger_ticks = 0
                self.captures += 1
                count += 1
            else:
                i += 1
        return count

    def _prey_step(self, pr: Prey):
        # flee + random
        # flee vector: away from wolf if near; else mild away if within radius
        wx, wy = self.wolf
        px, py = pr.pos
        dx = torus_delta(px, wx, self.n)
        dy = torus_delta(py, wy, self.n)
        ad = abs(dx) + abs(dy)
        if ad <= self.cfg["prey_flee_near"]:
            vx, vy = -np.sign(dx), -np.sign(dy)
        elif ad <= self.cfg["prey_flee_radius"]:
            vx, vy = -dx, -dy
            norm = max(1.0, math.hypot(vx, vy))
            vx, vy = vx / norm, vy / norm
        else:
            vx, vy = 0.0, 0.0
        # jitter
        if self.rng.random() < self.cfg["prey_flee_jitter_prob"]:
            vx += (self.rng.random() - 0.5) * self.cfg["prey_flee_eps"]
            vy += (self.rng.random() - 0.5) * self.cfg["prey_flee_eps"]
        # discretize to 8-dir or stay
        if abs(vx) + abs(vy) < 1e-6:
            cand = [STAY] + DIRS_8
            mv = cand[self.rng.integers(0, len(cand))]
        else:
            sx = 0 if abs(vx) < 1e-6 else (1 if vx > 0 else -1)
            sy = 0 if abs(vy) < 1e-6 else (1 if vy > 0 else -1)
            mv = (sx, sy)
        nx, ny = wrap_xy(px + mv[0], py + mv[1], self.n)
        if (nx, ny) != self.wolf:
            pr.pos = (nx, ny)

    # ------------- Gym API -------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.captures = 0
        self.hunger_ticks = 0
        # place wolf and preys
        pts = self._random_free_positions(1 + self.cfg["n_preys"])
        self.wolf = pts[0]
        self.preys = [Prey(p) for p in pts[1:]]
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        d_min, target = self._nearest_prey(self.wolf)
        if target is None:
            dx_norm = dy_norm = d_norm = 0.0
        else:
            dx = torus_delta(self.wolf[0], target[0], self.n)
            dy = torus_delta(self.wolf[1], target[1], self.n)
            dx_norm = dx / (self.n/2)  # in [-1,1]
            dy_norm = dy / (self.n/2)
            d_norm = d_min / self.n
        hunger_norm = clamp01(self.hunger_ticks / self.cfg["hunger_norm"])
        in_reach = 1.0 if self._prey_in_reach(self.wolf) else 0.0
        return np.array([hunger_norm, dx_norm, dy_norm, d_norm, in_reach], dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.steps += 1
    
        # --- d0 AVANT action (pour le shaping) ---
        d0, _ = self._nearest_prey(self.wolf)
    
        # faim (tick)
        self.hunger_ticks += 1
    
        # map action -> move (stay or sprint 2 cases)
        move = STAY if action == 0 else (2*DIRS_8[action-1][0], 2*DIRS_8[action-1][1])
    
        # --- appliquer le mouvement du loup, sous-pas de 1 case ---
        x, y = self.wolf
        captured = False
        for (sx, sy) in self._decompose(move):
            x, y = wrap_xy(x + sx, y + sy, self.n)
            self.wolf = (x, y)
    
            # tentative de capture AVANT le déplacement des proies
            if self._prey_in_reach(self.wolf):
                hn_before = clamp01(self.hunger_ticks / self.cfg["hunger_norm"])  # faim AVANT reset
                before = self.captures
                self._eat_here()           # remet hunger_ticks à 0
                if self.captures > before:
                    # créditer la reward de capture avec la faim avant reset
                    capture_reward = self.cfg["eat_reward_max"] * hn_before
                    captured = True
                    break  # on peut couper le sprint dès qu'on a mangé
    
        # faim bonus si sprint
        if action != 0:
            self.hunger_ticks += self.cfg["sprint_hunger_bonus"]
    
        # --- déplacement des proies (après la possibilité de manger) ---
        for pr in self.preys:
            if self.rng.random() < self.cfg["prey_move_prob"]:
                self._prey_step(pr)
    
        # --- reward de base ---
        reward = -self.cfg["step_penalty"]
        if action != 0:
            reward -= self.cfg["sprint_cost"]
        if captured:
            reward += capture_reward
    
        # --- d1 APRES transition (pour le shaping potentiel sûr) ---
        d1, _ = self._nearest_prey(self.wolf)
        if self.cfg.get("use_potential_shaping", False):
            alpha = float(self.cfg.get("potential_alpha", 0.1))
            gamma = float(self.cfg["gamma"])
            phi_s  = -alpha * (d0 / self.n)
            phi_sp = -alpha * (d1 / self.n)
            reward += gamma * phi_sp - phi_s
        
        # terminaisons
        terminated = False
        if self.captures >= self.cfg["capture_quota"]:
            terminated = True
        if self.hunger_ticks >= self.cfg["hunger_max"]:
            terminated = True
            reward -= 5.0  # fail penalty
        truncated = self.steps >= self.cfg["t_max"]
    
        obs = self._get_obs()
        info = {"captures": self.captures}
        return obs, reward, terminated, truncated, info


    # ------------- Rendering (optional) -------------
    def render(self):
        # Lazy import pygame only when needed to keep deps light for training headless
        try:
            import pygame
        except Exception:
            return None
        CELL = 10
        W = self.n * CELL
        H = self.n * CELL
        surf = pygame.Surface((W, H))
        surf.fill((25,25,28))

        # grid faint
        gridc = (38,38,44)
        for x in range(0, W, CELL):
            pygame.draw.line(surf, gridc, (x,0), (x,H))
        for y in range(0, H, CELL):
            pygame.draw.line(surf, gridc, (0,y), (W,y))

        # preys
        for pr in self.preys:
            rx = pr.pos[0]*CELL+2; ry = pr.pos[1]*CELL+2
            pygame.draw.rect(surf, (60,200,120), (rx,ry,CELL-4,CELL-4))
        # wolf
        wx = self.wolf[0]*CELL + CELL//2
        wy = self.wolf[1]*CELL + CELL//2
        pygame.draw.circle(surf, (255,170,0), (wx,wy), CELL//2-2)

        return surf
