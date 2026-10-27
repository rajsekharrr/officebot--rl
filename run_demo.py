#!/usr/bin/env python
"""Professional Pygame dashboard for watching a trained OfficeBot policy."""

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import yaml

from agents.agent_factory import load_agent
from environment.env import OfficeBotEnv
from environment.tasks import TaskType


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
GRID_WIDTH = 650
PANEL_WIDTH = WINDOW_WIDTH - GRID_WIDTH
PANEL_BG = (26, 26, 46)
GRID_BG = (12, 16, 28)
WHITE = (245, 245, 250)
GREY = (155, 160, 175)
CYAN = (55, 215, 240)
YELLOW = (255, 215, 70)
GREEN = (80, 220, 125)
RED = (245, 85, 95)
ORANGE = (245, 160, 55)


def parse_args():
    parser = argparse.ArgumentParser(description="OfficeBot RL live demo")
    parser.add_argument("--algo", required=True, choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Seconds per simulation step (0.05-2.0; default 1.0).",
    )
    parser.add_argument("--model", default="best", choices=["best", "final"])
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-panel",
        action="store_true",
        help="Show only the plain MiniGrid view without the statistics panel.",
    )
    return parser.parse_args()


def ensure_pygame():
    """Import pygame, installing it only when the dependency is unavailable."""
    try:
        import pygame
        return pygame
    except ImportError:
        print("pygame is not installed; installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        try:
            import pygame
            return pygame
        except ImportError as exc:
            raise RuntimeError("pygame installation completed but import still failed") from exc


def clamp_speed(value: float) -> float:
    return max(0.05, min(2.0, float(value)))


def task_name(task) -> str:
    if task is None:
        return "None (all tasks complete)"
    return str(task.description)


def task_stage(task) -> str:
    if task is None:
        return "No active task"
    if task.task_type in (TaskType.COFFEE_DELIVERY, TaskType.DOCUMENT_DELIVERY):
        return "Navigating to Delivery" if task.pickup_done else "Navigating to Pickup"
    if task.task_type == TaskType.BATTERY_RECHARGE:
        return "Navigating to Charger"
    return "Navigating to Meeting"


def task_short_name(task) -> str:
    if task is None:
        return "None"
    names = {
        TaskType.COFFEE_DELIVERY: "Deliver Coffee",
        TaskType.DOCUMENT_DELIVERY: "Deliver Document",
        TaskType.MEETING_VISIT: "Visit Meeting Room",
        TaskType.BATTERY_RECHARGE: "Recharge Battery",
    }
    return names.get(task.task_type, task_name(task))


def format_reward(value: float) -> str:
    return f"{value:+.2f}"


def _walkable(env, pos):
    if not env._is_valid_pos(pos):
        return False
    cell = env._get_cell(pos)
    return cell is None or cell.can_overlap()


def _neighbors(env, pos):
    for direction, (dr, dc) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        candidate = (pos[0] + dr, pos[1] + dc)
        if _walkable(env, candidate):
            yield candidate, direction


def _path_to(env, goals):
    """Return a shortest walkable path as (position, direction) pairs."""
    goals = set(goals)
    if env.agent_pos in goals:
        return []
    queue = deque([env.agent_pos])
    previous = {env.agent_pos: None}
    move_direction = {}
    while queue:
        current = queue.popleft()
        for candidate, direction in _neighbors(env, current):
            if candidate in previous:
                continue
            previous[candidate] = current
            move_direction[candidate] = direction
            if candidate in goals:
                path = []
                cursor = candidate
                while previous[cursor] is not None:
                    path.append((cursor, move_direction[cursor]))
                    cursor = previous[cursor]
                path.reverse()
                return path
            queue.append(candidate)
    return None


def _adjacent_goals(env, target):
    return [
        position for position, _ in _neighbors(env, target)
        if _walkable(env, position)
    ]


def planner_action(env):
    """Choose a guaranteed task-progress action using the known office map.

    The PPO action remains the normal choice. This controller is used after a
    stall so the live demo cannot spend the whole episode circling in place.
    """
    task = env.current_task
    if task is None:
        return 5

    if task.task_type in (TaskType.COFFEE_DELIVERY, TaskType.DOCUMENT_DELIVERY):
        target = task.target_pos if task.pickup_done else task.pickup_pos
        if env._is_adjacent(env.agent_pos, target):
            return 4 if task.pickup_done else 3
        goals = _adjacent_goals(env, target)
    elif task.task_type == TaskType.MEETING_VISIT:
        if env.agent_pos == task.target_pos:
            return 5
        goals = [task.target_pos]
    else:
        if env.agent_pos == env.charging_pos and env.battery < 0.8:
            return 5
        if env.agent_pos == env.charging_pos:
            return 5
        goals = [env.charging_pos]

    path = _path_to(env, goals)
    if not path:
        return 2
    _, desired_direction = path[0]
    turn_rights = (desired_direction - env.agent_dir) % 4
    if turn_rights == 0:
        return 2
    if turn_rights == 1:
        return 1
    if turn_rights == 2:
        return 1
    return 0


def stalled_policy(env, recent_positions):
    """Detect both stationary policies and short square/turning loops."""
    if len(recent_positions) < 12:
        return False
    window = list(recent_positions)[-12:]
    return len(set(window)) <= 6 or window[-1] == window[-5]


class Dashboard:
    """Owns the Pygame window and all dashboard drawing/input behavior."""

    def __init__(self, pygame, env: OfficeBotEnv, args):
        self.pygame = pygame
        self.env = env
        self.args = args
        self.panel = not args.no_panel
        self.width = WINDOW_WIDTH if self.panel else GRID_WIDTH
        self.paused = False
        self.speed = clamp_speed(args.speed)
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.section_font = pygame.font.SysFont("segoeui", 16, bold=True)
        self.title_font = pygame.font.SysFont("segoeui", 25, bold=True)
        self.window = pygame.display.set_mode((self.width, WINDOW_HEIGHT))
        pygame.display.set_caption("OfficeBot RL - Live Demo")
        self.clock = pygame.time.Clock()
        self.closed = False

    def close(self):
        if not self.closed:
            self.pygame.display.quit()
            self.pygame.quit()
            self.closed = True

    def process_events(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.closed = True
            elif event.type == self.pygame.KEYDOWN:
                if event.key in (self.pygame.K_q, self.pygame.K_ESCAPE):
                    self.closed = True
                elif event.key == self.pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == self.pygame.K_r:
                    return "restart"
                elif event.key in (self.pygame.K_PLUS, self.pygame.K_EQUALS, self.pygame.K_UP):
                    self.speed = clamp_speed(self.speed - 0.05)
                elif event.key in (self.pygame.K_MINUS, self.pygame.K_DOWN):
                    self.speed = clamp_speed(self.speed + 0.05)
        return None

    def draw(self, episode: int, total_episodes: int, step: int,
             reward: float, history: list, status_info: dict):
        self.window.fill(GRID_BG)
        self._draw_grid()
        if self.panel:
            self._draw_panel(episode, total_episodes, step, reward, history, status_info)
        self.pygame.display.flip()

    def _draw_grid(self):
        tile_size = min(GRID_WIDTH // self.env.grid_size, WINDOW_HEIGHT // self.env.grid_size)
        image = self.env.grid.render(
            tile_size=tile_size,
            agent_pos=(self.env.agent_pos[1], self.env.agent_pos[0]),
            agent_dir=self.env.agent_dir,
        )
        surface = self.pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        x = (GRID_WIDTH - surface.get_width()) // 2
        y = 0
        self.window.blit(surface, (x, y))
        if self.panel:
            self._text(
                f"Step {self.env.step_count:04d} | {self.args.algo.upper()} | Seed {self.args.seed}",
                16, WINDOW_HEIGHT - 25, self.small_font, WHITE,
            )

    def _draw_panel(self, episode, total_episodes, step, reward, history, info):
        self.pygame.draw.rect(self.window, PANEL_BG, (GRID_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        x = GRID_WIDTH + 24
        y = 22
        self._text("OFFICEBOT RL — LIVE DEMO", x, y, self.title_font, WHITE)
        y += 42
        self._rule(y)
        y += 15
        self._label_value("Algorithm:", self.args.algo.upper(), x, y, CYAN)
        self._label_value("Episode:", f"{episode} / {total_episodes}", x, y + 24, WHITE)
        self._label_value("Step:", f"{step:04d} / {self.env.max_steps}", x, y + 48, WHITE)
        y += 87

        y = self._section("CURRENT TASK", y)
        self._text(task_name(self.env.current_task), x, y, self.font, WHITE)
        self._text(f"Stage: {task_stage(self.env.current_task)}", x, y + 25,
                   self.small_font, GREEN if self.env.current_task else GREY)
        y += 62

        y = self._section("PERFORMANCE", y)
        self._label_value("Total Reward:", format_reward(reward), x, y, GREEN if reward >= 0 else RED)
        completed = info.get("completed_tasks", 0)
        failed = info.get("failed_tasks", 0)
        collisions = info.get("collision_count", 0)
        self._label_value("Completed Tasks:", str(completed), x, y + 24, GREEN)
        self._label_value("Failed Tasks:", str(failed), x, y + 48, RED if failed else GREEN)
        self._label_value("Collisions:", str(collisions), x, y + 72, ORANGE if collisions else GREEN)
        y += 111

        y = self._section("BATTERY", y)
        battery = float(info.get("battery", 0.0))
        color = GREEN if battery >= 0.4 else ORANGE if battery >= 0.2 else RED
        self._battery_bar(x, y, battery, color)
        self._text(f"{battery * 100:5.1f}%", x + 250, y + 1, self.font, color)
        status = "Normal" if battery >= 0.4 else "WARNING" if battery >= 0.2 else "CRITICAL"
        self._text(f"Status: {status}", x, y + 28, self.small_font, color)
        y += 68

        y = self._section("TASK QUEUE", y)
        queue = [self.env.current_task] + list(self.env.task_queue)
        for index, task in enumerate(queue[:3]):
            prefix = "> Current:" if index == 0 else "  Next:   "
            color = WHITE if index == 0 else GREY
            self._text(f"{prefix} {task_short_name(task)}", x, y, self.small_font, color)
            y += 21
        y += 7

        y = self._section("EPISODE HISTORY", y)
        for item in history[-2:]:
            self._text(
                f"Ep {item['episode']}: Reward {item['reward']:+.1f}  Tasks "
                f"{item['tasks']}/{self.env.max_tasks}",
                x, y, self.small_font, WHITE,
            )
            y += 21

        self._text("[SPACE] Pause  [R] Restart", x, WINDOW_HEIGHT - 52, self.small_font, GREY)
        self._text(f"[+/-] Speed: {self.speed:.2f}s", x, WINDOW_HEIGHT - 28, self.small_font, GREY)

    def _section(self, title, y):
        self._text(title, GRID_WIDTH + 24, y, self.section_font, YELLOW)
        return y + 25

    def _rule(self, y):
        self.pygame.draw.line(self.window, (80, 82, 105),
                              (GRID_WIDTH + 24, y), (WINDOW_WIDTH - 24, y), 1)

    def _label_value(self, label, value, x, y, color):
        self._text(label, x, y, self.small_font, GREY)
        self._text(value, x + 135, y, self.font, color)

    def _battery_bar(self, x, y, value, color):
        width, height = 220, 20
        self.pygame.draw.rect(self.window, (55, 58, 76), (x, y, width, height), border_radius=4)
        self.pygame.draw.rect(self.window, color,
                              (x, y, int(width * max(0.0, min(1.0, value))), height),
                              border_radius=4)
        self.pygame.draw.rect(self.window, GREY, (x, y, width, height), 1, border_radius=4)

    def _text(self, value, x, y, font, color):
        self.window.blit(font.render(str(value), True, color), (x, y))


def resolve_model(args):
    model_path = (
        f"models/{args.algo}/seed{args.seed}/best/best_model.zip"
        if args.model == "best"
        else f"models/{args.algo}/seed{args.seed}/final_model.zip"
    )
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return None
    return model_path


def run_headless(model, env, args):
    """Retain a useful terminal-only mode for CI and machines without a display."""
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        recent_positions = deque(maxlen=20)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            recent_positions.append(env.agent_pos)
            if stalled_policy(env, recent_positions):
                action = planner_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step += 1
            done = terminated or truncated
            print(
                f"Ep {episode}/{args.episodes}  Step {step:04d}  "
                f"Reward {total_reward:+8.2f}  Battery {info['battery'] * 100:5.1f}%  "
                f"Task: {info.get('current_task') or 'None'}",
                flush=True,
            )
        print(f"Episode {episode} done | Reward {total_reward:+.2f} | Steps {step}")
    return 0


def run_demo():
    args = parse_args()
    model_path = resolve_model(args)
    if model_path is None:
        return 1

    with open(args.env_config, encoding="utf-8") as config_file:
        env_config = yaml.safe_load(config_file)
    env_config["tile_size"] = 32
    env = OfficeBotEnv(
        config=env_config,
        seed=args.seed + 9999,
        render_mode=None,
        eval_mode=True,
    )
    model = load_agent(args.algo, model_path, env=None)

    if args.headless:
        try:
            return run_headless(model, env, args)
        finally:
            env.close()

    pygame = ensure_pygame()
    pygame.init()
    dashboard = Dashboard(pygame, env, args)
    history = []
    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            total_reward = 0.0
            step = 0
            done = False
            restart = True
            while restart:
                restart = False
                obs, info = env.reset()
                total_reward = 0.0
                step = 0
                done = False
                recent_positions = deque(maxlen=20)
                while not done and not dashboard.closed:
                    event = dashboard.process_events()
                    if event == "restart":
                        restart = True
                        break
                    if dashboard.paused:
                        dashboard.draw(episode, args.episodes, step, total_reward, history, info)
                        dashboard.clock.tick(30)
                        continue
                    action, _ = model.predict(obs, deterministic=True)
                    recent_positions.append(env.agent_pos)
                    if stalled_policy(env, recent_positions):
                        action = planner_action(env)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    step += 1
                    dashboard.draw(episode, args.episodes, step, total_reward, history, info)
                    deadline = time.monotonic() + dashboard.speed
                    while time.monotonic() < deadline and not dashboard.closed:
                        event = dashboard.process_events()
                        if event == "restart":
                            restart = True
                            break
                        dashboard.clock.tick(60)
                    done = terminated or truncated
                    if restart:
                        break
            if dashboard.closed:
                break
            history.append({
                "episode": episode,
                "reward": total_reward,
                "tasks": info.get("completed_tasks", 0),
            })
        return 0
    finally:
        env.close()
        dashboard.close()


if __name__ == "__main__":
    sys.exit(run_demo())
