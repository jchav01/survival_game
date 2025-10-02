# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 20:49:09 2025

@author: CES
"""

from __future__ import annotations

import sys
import pygame
from config import Config
from env_survival import SurvivalEnv, SurvivalEnvV2
from renderer import Renderer
from advanced_wolf_policy import choose_next_pos as advanced_wolf_policy
from results_logger import log_result
from wolf_policy import choose_next_pos as basic_wolf_policy


def _prompt_mode() -> str:
    print("Choisissez un mode:")
    print("  1 - Mode classique (env, politique avancée)")
    print("  2 - Mode avancé (env2, politique simple)")
    while True:
        choice = input("Votre choix (1/2): ").strip()
        if choice in {"1", "2"}:
            return choice
        print("Entrée invalide. Merci de répondre par 1 ou 2.")


def _build_game(cfg: Config):
    choice = _prompt_mode()
    if choice == "1":
        print("Mode 1 sélectionné: environnement classique avec politique avancée.")
        env = SurvivalEnv(cfg)
        policy = advanced_wolf_policy
    else:
        print("Mode 2 sélectionné: environnement avancé avec politique simple.")
        env = SurvivalEnvV2(cfg)
        policy = basic_wolf_policy
    return env, policy


def run():
    cfg = Config()
    env, policy = _build_game(cfg)
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
                next_pos = policy(env, env.wolf)
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