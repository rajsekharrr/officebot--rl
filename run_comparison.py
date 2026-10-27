#!/usr/bin/env python
"""Side-by-side random-policy versus trained PPO comparison demo."""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import yaml

from agents.agent_factory import load_agent
from environment.env import OfficeBotEnv


WIDTH, HEIGHT = 1400, 700
PANEL_BG = (26, 26, 46)
WHITE = (245, 245, 250)
GREY = (155, 160, 175)
CYAN = (55, 215, 240)
YELLOW = (255, 215, 70)
GREEN = (80, 220, 125)
RED = (245, 85, 95)
ORANGE = (245, 160, 55)


def parse_args():
    parser = argparse.ArgumentParser(description="OfficeBot policy comparison demo")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Seconds per step (0.05 to 2.0).")
    return parser.parse_args()


def pygame_module():
    try:
        import pygame
        return pygame
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        import pygame
        return pygame


def clamp_speed(speed):
    return max(0.05, min(2.0, float(speed)))


def reset_pair(random_env, trained_env, seed):
    random_obs, random_info = random_env.reset(seed=seed)
    trained_obs, trained_info = trained_env.reset(seed=seed)
    return random_obs, random_info, trained_obs, trained_info


def render_grid(pygame, env, width, height):
    tile = min(width // env.grid_size, height // env.grid_size)
    image = env.grid.render(
        tile_size=tile,
        agent_pos=(env.agent_pos[1], env.agent_pos[0]),
        agent_dir=env.agent_dir,
    )
    surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
    x = (width - surface.get_width()) // 2
    y = (height - surface.get_height()) // 2
    return surface, (x, y)


def text(screen, font, value, x, y, color=WHITE):
    screen.blit(font.render(str(value), True, color), (x, y))


def draw_agent_side(pygame, screen, env, title, subtitle, x, width, fonts,
                    reward, label, info):
    pygame.draw.rect(screen, PANEL_BG, (x, 0, width, HEIGHT))
    pygame.draw.line(screen, (80, 82, 105), (x, 0), (x, HEIGHT), 2)
    text(screen, fonts["title"], title, x + 24, 18)
    text(screen, fonts["subtitle"], subtitle, x + 24, 52, CYAN)
    grid_surface, position = render_grid(pygame, env, width - 32, 490)
    screen.blit(grid_surface, (x + 16 + position[0], 85 + position[1]))
    y = 585
    done = info.get("completed_tasks", 0)
    battery = float(info.get("battery", 0.0))
    battery_color = GREEN if battery > 0.4 else ORANGE if battery >= 0.2 else RED
    reward_color = GREEN if reward >= 0 else RED
    text(screen, fonts["body"], f"Total Reward: {reward:+.2f}", x + 24, y, reward_color)
    text(screen, fonts["body"], f"Tasks Done: {done}/{env.max_tasks}", x + 24, y + 29, GREEN)
    text(screen, fonts["body"], f"Battery: {battery * 100:.0f}%", x + 260, y, battery_color)
    text(screen, fonts["small"], f"Collisions: {info.get('collision_count', 0)}",
         x + 260, y + 31, ORANGE if info.get("collision_count", 0) else GREEN)
    text(screen, fonts["small"], f"Task: {label[:39]}", x + 24, y + 58, GREY)


def draw_overlay(pygame, screen, fonts, episode, random_result, trained_result):
    overlay = pygame.Surface((650, 310), pygame.SRCALPHA)
    overlay.fill((12, 12, 28, 245))
    left = (40, 35, 40)
    screen.blit(overlay, (375, 195))
    text(screen, fonts["title"], f"EPISODE {episode} RESULTS", 570, 220)
    text(screen, fonts["body"], "                 UNTRAINED       TRAINED", 430, 270, CYAN)
    rows = [
        ("Total Reward:", f"{random_result['reward']:+.1f}", f"{trained_result['reward']:+.1f}"),
        ("Tasks Done:", f"{random_result['tasks']}/5", f"{trained_result['tasks']}/5"),
        ("Collisions:", str(random_result["collisions"]), str(trained_result["collisions"])),
        ("Battery End:", f"{random_result['battery']:.0f}%", f"{trained_result['battery']:.0f}%"),
        ("Survived:", "YES" if random_result["survived"] else "NO",
                     "YES" if trained_result["survived"] else "NO"),
    ]
    for row, values in enumerate(rows):
        y = 310 + row * 28
        text(screen, fonts["small"], values[0], 430, y, GREY)
        text(screen, fonts["body"], values[1], 640, y,
             GREEN if random_result["survived"] else RED)
        text(screen, fonts["body"], values[2], 815, y,
             GREEN if trained_result["survived"] else RED)
    winner = trained_result["reward"] - random_result["reward"]
    text(screen, fonts["body"], f"Trained agent won by {winner:+.1f} pts", 555, 475,
         GREEN if winner >= 0 else RED)


def result(env, reward, info):
    return {
        "reward": reward,
        "tasks": info.get("completed_tasks", 0),
        "collisions": info.get("collision_count", 0),
        "battery": info.get("battery", 0.0) * 100,
        "survived": info.get("termination_reason") != "battery_dead",
    }


def main():
    args = parse_args()
    model_path = f"models/{args.algo}/seed{args.seed}/best/best_model.zip"
    if not os.path.exists(model_path):
        print(f"Error: trained model not found: {model_path}")
        return 1

    with open("configs/env_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["tile_size"] = 32
    random_env = OfficeBotEnv(config=config, seed=args.seed, eval_mode=True)
    trained_env = OfficeBotEnv(config=config, seed=args.seed, eval_mode=True)
    model = load_agent(args.algo, model_path, env=None)
    pygame = pygame_module()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("OfficeBot RL - Random vs Trained")
    fonts = {
        "title": pygame.font.SysFont("segoeui", 25, bold=True),
        "subtitle": pygame.font.SysFont("segoeui", 18, bold=True),
        "body": pygame.font.SysFont("segoeui", 17),
        "small": pygame.font.SysFont("segoeui", 14),
    }
    clock = pygame.time.Clock()
    speed = clamp_speed(args.speed)
    paused = False
    quit_requested = False
    try:
        for episode in range(1, args.episodes + 1):
            random_obs, random_info, trained_obs, trained_info = reset_pair(
                random_env, trained_env, args.seed + episode - 1
            )
            random_done = trained_done = False
            random_reward = trained_reward = 0.0
            step = 0
            episode_restart = True
            while episode_restart and not quit_requested:
                episode_restart = False
                while not (random_done and trained_done) and not quit_requested:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit_requested = True
                        elif event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                quit_requested = True
                            elif event.key == pygame.K_SPACE:
                                paused = not paused
                            elif event.key == pygame.K_r:
                                episode_restart = True
                                break
                            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
                                speed = clamp_speed(speed - 0.05)
                            elif event.key in (pygame.K_MINUS, pygame.K_DOWN):
                                speed = clamp_speed(speed + 0.05)
                    if quit_requested or episode_restart:
                        break
                    if not paused:
                        if not random_done:
                            random_action = random_env.action_space.sample()
                            random_obs, random_step_reward, term, trunc, random_info = random_env.step(
                                random_action
                            )
                            random_reward += float(random_step_reward)
                            random_done = term or trunc
                        if not trained_done:
                            trained_action, _ = model.predict(trained_obs, deterministic=True)
                            trained_obs, trained_step_reward, term, trunc, trained_info = trained_env.step(
                                trained_action
                            )
                            trained_reward += float(trained_step_reward)
                            trained_done = term or trunc
                        step += 1
                    screen.fill((12, 16, 28))
                    draw_agent_side(
                        pygame, screen, random_env, "UNTRAINED AGENT", "(Random Policy)",
                        0, WIDTH // 2, fonts, random_reward, random_info.get("current_task"), random_info
                    )
                    draw_agent_side(
                        pygame, screen, trained_env, "TRAINED AGENT",
                        f"({args.algo.upper()} 500k steps)", WIDTH // 2, WIDTH // 2,
                        fonts, trained_reward, trained_info.get("current_task"), trained_info
                    )
                    text(screen, fonts["body"], f"VS — Step {step:04d} — "
                         f"{'[SPACE] Resume' if paused else '[SPACE] Pause'}", 540, 665, WHITE)
                    text(screen, fonts["small"], f"Speed: {speed:.2f}s  [+/-] adjust  [Q] quit",
                         1000, 665, GREY)
                    pygame.display.flip()
                    if not paused:
                        deadline = time.monotonic() + speed
                        while time.monotonic() < deadline:
                            pygame.event.pump()
                            clock.tick(60)
                if episode_restart and not quit_requested:
                    random_obs, random_info, trained_obs, trained_info = reset_pair(
                        random_env, trained_env, args.seed + episode - 1
                    )
                    random_done = trained_done = False
                    random_reward = trained_reward = 0.0
                    step = 0
            if quit_requested:
                break
            random_result = result(random_env, random_reward, random_info)
            trained_result = result(trained_env, trained_reward, trained_info)
            overlay_until = time.monotonic() + 3.0
            while time.monotonic() < overlay_until and not quit_requested:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_q, pygame.K_ESCAPE
                    ):
                        quit_requested = True
                draw_overlay(pygame, screen, fonts, episode, random_result, trained_result)
                pygame.display.flip()
                clock.tick(60)
    finally:
        random_env.close()
        trained_env.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
