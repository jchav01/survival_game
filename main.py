# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 20:49:09 2025

@author: CES
"""

from __future__ import annotations

import sys
import pygame
from config import Config
from env_survival import SurvivalEnv
from renderer import Renderer
from advanced_wolf_policy import choose_next_pos
from results_logger import log_result


def run():
    cfg = Config()
    env = SurvivalEnv(cfg)
    env.reset()
    rnd = Renderer(env, cfg)

    paused = False

    try:
        while True:
            # --- events ---
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return
                    if e.key == pygame.K_SPACE:
                        paused = not paused

            if not paused:
                # dynamique du loup (externe)
                next_pos = choose_next_pos(env, env.wolf)
                env.set_wolf(next_pos)

                # step environnement
                done = env.step()
                if done:
                    # petit écran de fin + sauvegarde des stats
                    log_result(env.score, env.tick, cfg.FPS)
                    print(f"[GAME OVER] ticks={env.tick} score={env.score}")
                    pygame.time.wait(800)
                    return

            # rendu
            rnd.draw()
            rnd.tick()
    finally:
        rnd.close()


if __name__ == "__main__":
    run()