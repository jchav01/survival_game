# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 20:48:40 2025

@author: CES
"""

from __future__ import annotations
from typing import Tuple
from env_survival import SurvivalEnv, cheb

Pos = Tuple[int,int]

# === API à implémenter/itérer par toi ===
# Retourne la prochaine position du loup (bornée par l'env).

def choose_next_pos(env: SurvivalEnv, prev_pos: Pos) -> Pos:
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