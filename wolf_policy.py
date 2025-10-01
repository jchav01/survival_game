# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 20:48:40 2025

@author: CES
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

from preys import Pos


class SupportsWolfEnv(Protocol):
    n: int

    def nearest_prey(self, pos: Optional[Pos] = None) -> Tuple[int, Optional[Pos]]:
        ...


def choose_next_pos(env: SupportsWolfEnv, prev_pos: Pos) -> Pos:
    """Politique placeholder : aller (bêtement) vers la proie la plus proche.
    Remplace par ta dynamique (RL, heuristique, etc.).
    """
    n = env.n
    wolf = prev_pos
    d, p = env.nearest_prey(wolf)
    if p is None:
        return wolf  # rien à faire
    dx = 0 if p[0] == wolf[0] else (1 if p[0] > wolf[0] else -1)
    dy = 0 if p[1] == wolf[1] else (1 if p[1] > wolf[1] else -1)
    nx = max(0, min(n-1, wolf[0] + dx))
    ny = max(0, min(n-1, wolf[1] + dy))
    return (nx, ny)