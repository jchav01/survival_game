# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 20:48:14 2025

@author: CES
"""

from __future__ import annotations
import pygame
from typing import Protocol, Sequence, Tuple

from config import Config

Color = Tuple[int, int, int]
Pos = Tuple[int, int]


class SupportsRenderEnv(Protocol):
    n: int
    preys: Sequence
    wolf: Pos
    hp: float
    tick: int
    score: int


class Renderer:
    def __init__(self, env: SupportsRenderEnv, cfg: Config):
        self.env = env
        self.cfg = cfg
        pygame.init()
        self.cell = cfg.CELL
        self.w = env.n * self.cell

        self.h = env.n * self.cell + 70  # bandeau HUD élargi pour les infos

        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Survival Wolf")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

    def draw(self):
        scr = self.screen
        scr.fill((20,20,24))

        # grille
        if self.cfg.SHOW_GRID:
            gc = (36,36,42)
            for x in range(0, self.env.n * self.cell, self.cell):
                pygame.draw.line(scr, gc, (x,0), (x, self.env.n*self.cell))
            for y in range(0, self.env.n * self.cell, self.cell):
                pygame.draw.line(scr, gc, (0,y), (self.env.n*self.cell, y))

        # proies
        for pr in self.env.preys:
            x, y = pr.pos
            pygame.draw.rect(scr, (70,200,120), (x*self.cell+2, y*self.cell+2, self.cell-4, self.cell-4))

        # loup
        wx, wy = self.env.wolf
        pygame.draw.circle(scr, (255,170,0), (wx*self.cell + self.cell//2, wy*self.cell + self.cell//2), self.cell//2-2)

        # bords
        pygame.draw.rect(scr, (90,90,100), (0,0,self.env.n*self.cell-1,self.env.n*self.cell-1), width=2)

        # HUD
        hud_y = self.env.n * self.cell

        hud_height = 70
        stats_height = 26
        bar_height = 20
        pygame.draw.rect(scr, (16,16,18), (0, hud_y, self.w, hud_height))

        # zone texte au-dessus de la barre de vie
        next_spawn_tick = getattr(self.env, "_next_spawn_tick", 0)
        surv_seconds = self.env.tick / self.cfg.FPS if self.cfg.FPS else float(self.env.tick)
        txt = (
            f"Score: {getattr(self.env, 'score', 0)}    "
            f"Survie: {surv_seconds:5.1f}s    "
            f"Tick: {self.env.tick}    "
            f"Next spawn: {next_spawn_tick}"
        )
        surf = self.font.render(txt, True, (220,220,230))
        scr.blit(surf, (12, hud_y + 6))

        # HP bar
        hp_frac = 0.0 if self.cfg.HP_MAX == 0 else self.env.hp / self.cfg.HP_MAX
        bar_y = hud_y + stats_height + 10
        pygame.draw.rect(scr, (60,60,70), (10, bar_y, self.w-20, bar_height))
        pygame.draw.rect(scr, (220,120,40), (10, bar_y, int((self.w-20) * hp_frac), bar_height))
        hp_txt = f"HP: {self.env.hp:.0f}/{self.cfg.HP_MAX}"
        hp_surf = self.font.render(hp_txt, True, (15,15,18))
        scr.blit(hp_surf, (14, bar_y + 2))


        pygame.display.flip()

    def tick(self):
        self.clock.tick(self.cfg.FPS)

    def close(self):
        pygame.quit()

