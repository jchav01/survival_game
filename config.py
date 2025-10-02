from dataclasses import dataclass


@dataclass
class Config:
    GRID_SIZE: int = 60
    N_PREYS_START: int = 21
    PREY_MOVE_PROB: float = 0.80  # proba de bouger lors des ticks actifs
    PREY_MOVES_EVERY_OTHER_TICK: bool = True  # proies ne bougent qu’un tick sur deux

    HP_MAX: int = 100
    HP_DECAY_PER_TICK: float = 0.8
    HP_DECAY_STATIC: float = 0.4
    HP_DECAY_MOVING: float = 0.8
    EAT_HEAL_MISSING_FRACTION: float = 0.6  # portion de la vie manquante rendue en mangeant

    SPAWN_INTERVAL_START: int = 25  # ticks entre spawns au début
    SPAWN_INTERVAL_GROWTH: float = 1.1  # facteur multiplicatif après chaque spawn
    MAX_PREYS: int = 80

    TTL_STATIC: int = 120
    TTL_ERRATIC: int = 80
    TTL_FLEE: int = 50
    FLEE_MOVE_PERIOD: int = 5
    FLEE_MOVE_REST_PHASE: int = 0
    FLEE_RADIUS: int = 8

    RNG_SEED: int | None = None

    # Rendu
    CELL: int = 12
    SHOW_GRID: bool = True
    FPS: int = 30