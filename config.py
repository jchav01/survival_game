## config.py

from dataclasses import dataclass


@dataclass
class Config:
    GRID_SIZE: int = 60
    N_PREYS_START: int = 12
    PREY_MOVE_PROB: float = 0.80 # proba de bouger lors des ticks actifs
    PREY_MOVES_EVERY_OTHER_TICK: bool = True # proies ne bougent qu’un tick sur deux
    
    
    HP_MAX: int = 100
    HP_DECAY_PER_TICK: float = 0.25
    EAT_HEAL: int = 25
    
    
    SPAWN_INTERVAL_START: int = 20 # ticks entre spawns au début
    SPAWN_INTERVAL_GROWTH: float = 1.15 # facteur multiplicatif après chaque spawn
    MAX_PREYS: int = 80
    
    
    RNG_SEED: int | None = None
    
    
    # Rendu
    CELL: int = 12
    SHOW_GRID: bool = True
    FPS: int = 30