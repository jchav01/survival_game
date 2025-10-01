from __future__ import annotations

import math
from typing import Tuple

from env_survival import SurvivalEnv

Pos = Tuple[int, int]


def _estimate_travel_ticks(distance: int, env: SurvivalEnv) -> int:
    """Nombre de ticks attendus pour rejoindre une proie à distance donnée."""
    if distance <= 1:
        return 1

    cfg = env.cfg
    base_ticks = distance - 1
    move_activity = cfg.PREY_MOVE_PROB
    if getattr(cfg, "PREY_MOVES_EVERY_OTHER_TICK", False):
        move_activity *= 0.5

    effective_progress = max(0.2, 1.0 - 0.5 * move_activity)
    return max(1, math.ceil(base_ticks / effective_progress))


def hunt_threshold(distance: int, env: SurvivalEnv) -> float:
    """HP maximale à laquelle on commence à chasser une proie donnée."""
    travel_ticks = _estimate_travel_ticks(distance, env)
    decay = env.cfg.HP_DECAY_PER_TICK
    safety_hp = max(2.0, 3.0 * decay)

    threshold = safety_hp + travel_ticks * decay

    max_hunt_hp = float(env.cfg.HP_MAX) - _post_hunt_buffer(env)
    return max(safety_hp, min(max_hunt_hp, threshold))


def _post_hunt_buffer(env: SurvivalEnv) -> float:
    """Quantité de HP à conserver pour ne pas gaspiller le soin attendu."""
    cfg = env.cfg
    if hasattr(cfg, "EAT_HEAL_MISSING_FRACTION"):
        fraction = float(getattr(cfg, "EAT_HEAL_MISSING_FRACTION"))
        if 0.0 < fraction < 1.0:
            # On cherche à arriver avec peu de vie pour bénéficier du pourcentage.
            return max(0.0, (1.0 - fraction) * float(cfg.HP_MAX))
    if hasattr(cfg, "EAT_HEAL"):
        heal = float(getattr(cfg, "EAT_HEAL"))
        return max(0.0, heal)
    return 0.0


def _step_towards(pos: Pos, target: Pos, n: int) -> Pos:
    dx = 0 if target[0] == pos[0] else (1 if target[0] > pos[0] else -1)
    dy = 0 if target[1] == pos[1] else (1 if target[1] > pos[1] else -1)
    return (max(0, min(n - 1, pos[0] + dx)), max(0, min(n - 1, pos[1] + dy)))


def _step_away(pos: Pos, target: Pos, n: int) -> Pos:
    dx = 0 if target[0] == pos[0] else (-1 if target[0] > pos[0] else 1)
    dy = 0 if target[1] == pos[1] else (-1 if target[1] > pos[1] else 1)
    candidate = (pos[0] + dx, pos[1] + dy)
    if 0 <= candidate[0] < n and 0 <= candidate[1] < n:
        return candidate
    return pos


def choose_next_pos(env: SurvivalEnv, prev_pos: Pos) -> Pos:
    """Politique presque-optimale : chasse sur seuil, sinon temporise."""
    distance, prey = env.nearest_prey(prev_pos)
    if prey is None:
        return prev_pos

    threshold = hunt_threshold(distance, env)
    if env.hp > threshold:
        # On évite de manger trop tôt, surtout si une proie est collée.
        if distance <= 1:
            return _step_away(prev_pos, prey, env.n)
        return prev_pos

    # En dessous du seuil : cap sur la proie la plus proche.
    return _step_towards(prev_pos, prey, env.n)