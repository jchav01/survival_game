# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 20:48:14 2025

@author: CES
"""

from __future__ import annotations
import pygame
from typing import Tuple
from config import Config
from env_survival import SurvivalEnv

Color = Tuple[int,int,int]

class Renderer:
    def __init__(self, env: SurvivalEnv, cfg: Config):
        self.env = env
        self.cfg = cfg
        pygame.init()
        self.cell = cfg.CELL
        self.w = env.n * self.cell
        self.h = env.n * self.cell + 40  # bandeau HUD
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
        pygame.draw.rect(scr, (16,16,18), (0, hud_y, self.w, 40))
        # HP bar
        hp_frac = self.env.hp / self.cfg.HP_MAX
        pygame.draw.rect(scr, (60,60,70), (10, hud_y+10, self.w-20, 20))
        pygame.draw.rect(scr, (220,120,40), (10, hud_y+10, int((self.w-20) * hp_frac), 20))
        # texte
        txt = f"tick={self.env.tick}  hp={self.env.hp}/{self.cfg.HP_MAX}  preys={len(self.env.preys)}  next_spawn≈{getattr(self.env, '_next_spawn_tick', 0)}"
        surf = self.font.render(txt, True, (220,220,230))
        scr.blit(surf, (12, hud_y+10))

        pygame.display.flip()

    def tick(self):
        self.clock.tick(self.cfg.FPS)

    def close(self):
        pygame.quit()