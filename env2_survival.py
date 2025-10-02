# env2_survival.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from config import Config

Pos = Tuple[int, int]
DIRS_8 = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
STAY = (0,0)

# Types de proies
PREY_STATIC  = 0  # +1 pt
PREY_ERRATIC = 1  # +2 pts
PREY_FLEE    = 2  # +3 pts

PREY_POINTS = {
    PREY_STATIC: 1,
    PREY_ERRATIC: 2,
    PREY_FLEE: 3,
}

# Cadence des fuyantes : bougent 4 ticks sur 5
FLEE_MOVE_PERIOD = 5        # période de 5 ticks
FLEE_MOVE_REST_PHASE = 0    # repos quand (tick % 5) == 0  -> 4/5 actifs

FLEE_RADIUS = 8  # distance de Tchebychev pour déclencher la fuite


def in_bounds(p: Pos, n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))


def sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


@dataclass
class Prey:
    pos: Pos
    ptype: int     # PREY_STATIC / PREY_ERRATIC / PREY_FLEE
    ttl: int       # ticks restants avant disparition


class SurvivalEnv:
    """
    env2 — ajout TTL et double décay HP (repos vs mouvement).

    - 3 types de proies (1/3 chacun, spawns cycliques):
        STATIC (+1 pt)    : ne bouge pas.
        ERRATIC (+2 pts)  : marche aléatoire (stay+8 voisins).
        FLEE (+3 pts)     : comme ERRATIC tant que le loup est loin ; si dist <= 6, fuit (1 pas)
                            à l'opposé ; si bord, dévie perpendiculairement (choix aléatoire si besoin).
    - TTL : chaque proie a un temps de vie (plus court pour +3). À 0 -> disparition.
    - HP decay : deux constantes
        * HP_DECAY_STATIC : quand le loup NE bouge PAS ce tick
        * HP_DECAY_MOVING : quand le loup a bougé ce tick
      (fallback si non présents: on dérive depuis HP_DECAY_PER_TICK.)
    - Score : +1/+2/+3 à la capture selon le type.
    - Heal : identique (via EAT_HEAL_MISSING_FRACTION).
    - Mouvement des proies : respecte PREY_MOVES_EVERY_OTHER_TICK et PREY_MOVE_PROB.
    - La dynamique du loup est externe: appeler set_wolf() avant step().
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n = cfg.GRID_SIZE
        self.rng = np.random.default_rng(cfg.RNG_SEED)

        self.tick: int = 0
        self.wolf: Pos = (0, 0)
        self._wolf_prev: Pos = (0, 0)  # pour savoir si le loup a bougé ce tick
        self.hp: float = float(cfg.HP_MAX)
        self.preys: List[Prey] = []
        self._ate_this_tick: bool = False

        self.score: int = 0

        # Spawn cyclique
        self._spawn_cycle = [PREY_STATIC, PREY_ERRATIC, PREY_FLEE]
        self._spawn_idx = 0

        # Scheduler de spawn (délai croissant)
        self._spawn_interval: float = float(cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick: int = int(self._spawn_interval)

        # TTL par type (fallback si non défini dans Config)
        self._ttl_static  = int(getattr(cfg, "TTL_STATIC", 120))
        self._ttl_erratic = int(getattr(cfg, "TTL_ERRATIC", 80))
        self._ttl_flee    = int(getattr(cfg, "TTL_FLEE", 50))

        # Décay HP (fallbacks)
        base_decay = float(getattr(cfg, "HP_DECAY_PER_TICK", 0.35))
        self._decay_static = float(getattr(cfg, "HP_DECAY_STATIC", base_decay * 0.5))
        self._decay_moving = float(getattr(cfg, "HP_DECAY_MOVING", base_decay))

    # -------- API externe (contrôle du loup) --------
    def set_wolf(self, pos: Pos):
        x = max(0, min(self.n - 1, int(pos[0])))
        y = max(0, min(self.n - 1, int(pos[1])))
        self.wolf = (x, y)

    # -------- Reset --------
    def reset(self):
        self.tick = 0
        self.hp = float(self.cfg.HP_MAX)
        self.preys = []
        self._ate_this_tick = False
        self.score = 0

        # place wolf
        self.wolf = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))
        self._wolf_prev = self.wolf

        # place initial preys (en respectant l'ordre cyclique)
        occ = {self.wolf}
        for _ in range(self.cfg.N_PREYS_START):
            p = self._sample_free_cell(occ)
            if p is None:
                break
            ptype = self._spawn_cycle[self._spawn_idx]
            self._spawn_idx = (self._spawn_idx + 1) % len(self._spawn_cycle)
            ttl = self._initial_ttl_for(ptype)
            self.preys.append(Prey(p, ptype, ttl))
            occ.add(p)

        # reset spawn schedule
        self._spawn_interval = float(self.cfg.SPAWN_INTERVAL_START)
        self._next_spawn_tick = int(self._spawn_interval)

    # -------- Helpers --------
    def _initial_ttl_for(self, ptype: int) -> int:
        if ptype == PREY_STATIC:
            return self._ttl_static
        elif ptype == PREY_ERRATIC:
            return self._ttl_erratic
        else:
            return self._ttl_flee

    def _sample_free_cell(self, occ: set) -> Optional[Pos]:
        tries = 0
        while tries < 400:
            tries += 1
            p = (self.rng.integers(0, self.n), self.rng.integers(0, self.n))
            if p not in occ:
                return p
        return None

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
                best_d, best_p = d, pr.pos
        return best_d, best_p

    # -------- Eating --------
    def _apply_heal(self):
        missing = float(self.cfg.HP_MAX) - self.hp
        if missing <= 0:
            return
        heal = missing * float(self.cfg.EAT_HEAL_MISSING_FRACTION)
        if heal <= 0:
            return
        self.hp = min(float(self.cfg.HP_MAX), self.hp + heal)

    def _eat_if_in_reach(self):
        """Mange au plus une proie adjacente (Chebyshev <= 1), ajoute le score selon le type."""
        if self._ate_this_tick:
            return
        for idx, prey in enumerate(self.preys):
            if cheb(self.wolf, prey.pos) <= 1:
                # retire la proie
                self.preys.pop(idx)
                # heal
                self._apply_heal()
                # score
                self.score += PREY_POINTS.get(prey.ptype, 1)
                self._ate_this_tick = True
                break

    # -------- Mouvement des proies --------
    def _preys_step(self):
        wolf_cell = self.wolf
        n = self.n
    
        # Occupation actuelle
        occupied = {pr.pos for pr in self.preys}
        order = list(range(len(self.preys)))
        self.rng.shuffle(order)
    
        for idx in order:
            pr = self.preys[idx]
            start = pr.pos
    
            if pr.ptype == PREY_STATIC:
                # ne bouge pas
                continue
    
            # --- Cadence de mouvement selon le type ---
            if pr.ptype == PREY_ERRATIC:
                # erratiques : respectent éventuellement le mode "1 tick sur 2"
                if self.cfg.PREY_MOVES_EVERY_OTHER_TICK and (self.tick % 2 == 1):
                    continue
            else:
                # fuyantes : bougent 4/5 (repos sur les ticks où (tick % 5) == FLEE_MOVE_REST_PHASE)
                if (self.tick % FLEE_MOVE_PERIOD) == FLEE_MOVE_REST_PHASE:
                    continue
    
            # Probabilité de bouger (commune erratique/fuyante)
            if self.rng.random() >= self.cfg.PREY_MOVE_PROB:
                continue
    
            # On libère sa case de départ (on va tenter de bouger)
            occupied.discard(start)
    
            if pr.ptype == PREY_ERRATIC:
                new_pos = self._step_erratic(start, wolf_cell, occupied, n)
            else:  # PREY_FLEE
                if cheb(start, wolf_cell) <= FLEE_RADIUS:
                    new_pos = self._step_flee(start, wolf_cell, occupied, n)
                else:
                    # hors rayon: se comporte comme ERRATIC
                    new_pos = self._step_erratic(start, wolf_cell, occupied, n)
    
            # Appliquer le mouvement
            if new_pos is None:
                # pas de case dispo -> rester sur place si possible
                if start != wolf_cell and start not in occupied:
                    pr.pos = start
                    occupied.add(start)
                else:
                    # fallback: essayer une case voisine libre quelconque
                    candidates = [(start[0]+dx, start[1]+dy) for dx,dy in ([STAY]+DIRS_8)]
                    candidates = [c for c in candidates if in_bounds(c, n) and c != wolf_cell and c not in occupied]
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


    def _step_erratic(self, start: Pos, wolf: Pos, occupied: set, n: int) -> Optional[Pos]:
        # Random parmi stay+8 voisins, borné, sans collision, pas sur le loup
        candidates = [(start[0]+dx, start[1]+dy) for dx,dy in ([STAY] + DIRS_8)]
        candidates = [c for c in candidates if in_bounds(c, n) and c != wolf and c not in occupied]
        if not candidates:
            return None
        return candidates[self.rng.integers(0, len(candidates))]

    def _step_flee(self, start: Pos, wolf: Pos, occupied: set, n: int) -> Optional[Pos]:
        # Vecteur de fuite (à l'opposé du loup), 1 pas
        vx = sign(start[0] - wolf[0])
        vy = sign(start[1] - wolf[1])
        if vx == 0 and vy == 0:
            return self._step_erratic(start, wolf, occupied, n)

        primary = (start[0] + vx, start[1] + vy)
        if in_bounds(primary, n) and primary not in occupied and primary != wolf:
            return primary

        # Déviation perpendiculaire si bord
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

        # Sinon: essayer d'augmenter la distance
        better = []
        same = []
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

    # -------- TTL & Spawns --------
    def _decrement_and_purge_ttl(self):
        """Décrémente le TTL de chaque proie et supprime celles qui expirent avant toute autre action."""
        i = 0
        while i < len(self.preys):
            self.preys[i].ttl -= 1
            if self.preys[i].ttl <= 0:
                self.preys.pop(i)
            else:
                i += 1

    def _maybe_spawn(self):
        if len(self.preys) >= self.cfg.MAX_PREYS:
            return
        if self.tick >= self._next_spawn_tick:
            occ = {self.wolf, *(pr.pos for pr in self.preys)}
            p = self._sample_free_cell(occ)
            if p is not None:
                ptype = self._spawn_cycle[self._spawn_idx]
                self._spawn_idx = (self._spawn_idx + 1) % len(self._spawn_cycle)
                ttl = self._initial_ttl_for(ptype)
                self.preys.append(Prey(p, ptype, ttl))
            # planifier prochain spawn (délai croissant)
            self._spawn_interval = max(1.0, self._spawn_interval * self.cfg.SPAWN_INTERVAL_GROWTH)
            self._next_spawn_tick = self.tick + int(round(self._spawn_interval))

    # -------- Step --------
    def step(self):
        self.tick += 1
        self._ate_this_tick = False

        # 0) TTL : décrémente et purge d'abord
        self._decrement_and_purge_ttl()

        # 1) Décay HP en fonction du mouvement ce tick
        moved = (self.wolf != self._wolf_prev)
        self.hp = max(0.0, self.hp - (self._decay_moving if moved else self._decay_static))
        self._wolf_prev = self.wolf  # mémoriser la position utilisée ce tick

        # 2) Manger AVANT déplacement (si encore vivant)
        if self.hp > 0.0:
            self._eat_if_in_reach()

        # 3) Déplacement des proies
        self._preys_step()

        # 4) Spawns
        self._maybe_spawn()

        # 5) Manger APRES déplacement
        if self.hp > 0.0:
            self._eat_if_in_reach()

        # 6) Fin si HP épuisés
        done = (self.hp <= 0.0)
        return done
