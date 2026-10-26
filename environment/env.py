# environment/env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Floor

from environment.constants import (
    GRID_SIZE,
    COFFEE_MACHINE_POS,
    PRINTER_POS,
    CHARGING_STATION_POS,
    DESK_POSITIONS,
    AGENT_START_POS,
    MEETING_ROOM_POS,
    MEETING_TABLE_POS,
    DIR_TO_DELTA,
)
from environment.objects import (
    CoffeeMachine,
    Printer,
    ChargingStation,
    Desk,
    MeetingTable,
)
from environment.tasks import TaskQueueManager, TaskType, Task


class OfficeBotEnv(gym.Env):
    """
    Custom office environment for OfficeBot RL.

    Implements Gymnasium API (reset, step, render, close).
    Does NOT subclass MiniGridEnv directly — builds a custom Grid
    and renders it using MiniGrid's Grid.render() utility.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, config: dict, seed: int = 42,
                 render_mode: str = None, eval_mode: bool = False):
        super().__init__()
        self.config = config
        self.seed_val = seed
        self.render_mode = render_mode
        self.eval_mode = eval_mode

        # Grid dimensions
        self.grid_size = config['grid_size']  # 15
        self.max_steps = config['max_steps']  # 500
        self.max_tasks = config['max_tasks']  # 5

        # Fixed positions (from constants.py)
        self.coffee_machine_pos = COFFEE_MACHINE_POS
        self.printer_pos        = PRINTER_POS
        self.charging_pos       = CHARGING_STATION_POS
        self.desk_positions     = DESK_POSITIONS
        self.agent_start_pos    = AGENT_START_POS
        self.meeting_room_pos   = MEETING_ROOM_POS

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        # RNG
        self.rng = np.random.default_rng(seed)

        # State (initialised in reset())
        self.grid         = None
        self.agent_pos    = None
        self.agent_dir    = 0
        self.battery      = 1.0
        self.carrying_item = None
        self.task_queue   = []
        self.current_task = None
        self.completed_tasks = []
        self.all_tasks    = []
        self.step_count   = 0
        self.collision_count = 0

        # Rendering
        self.window = None
        self.clock  = None

    def reset(self, seed=None, options=None):
        """Reset environment to start of new episode."""
        # Call parent reset to handle seed properly
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Build the grid
        self.grid = self._build_grid()

        # Place agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = 0

        # Reset state
        self.battery       = self.config['battery_initial']
        self.carrying_item = None
        self.step_count    = 0
        self.collision_count = 0
        self.completed_tasks = []

        # Generate tasks
        manager = TaskQueueManager(
            max_tasks=self.max_tasks,
            desk_positions=self.desk_positions,
        )
        self.all_tasks    = manager.generate_episode_tasks(self.rng)
        self.task_queue   = list(self.all_tasks)
        self.current_task = self.task_queue.pop(0) if self.task_queue else None

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        """Execute one action and return (obs, reward, terminated, truncated, info)."""
        assert self.grid is not None, "Call reset() before step()."

        reward = self.config['rewards']['step_penalty']
        terminated = False
        truncated  = False

        prev_pos = self.agent_pos

        # Update battery FIRST (drains on every step, including recharge attempts)
        battery_reward = self._update_battery(action)
        reward += battery_reward

        # Check termination EARLY - if battery is dead, end immediately
        if self.battery <= 0.0:
            terminated = True
            reason = 'battery_dead'
            reward += self.config['rewards']['battery_depleted']
            obs  = self._get_obs()
            info = self._get_info(termination_reason=reason)
            return obs, reward, terminated, False, info

        # Execute action
        if action == 0:    # TURN_LEFT
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:  # TURN_RIGHT
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:  # MOVE_FORWARD
            reward += self._try_move_forward()
        elif action == 3:  # PICKUP
            reward += self._try_pickup()
        elif action == 4:  # DELIVER
            reward += self._try_deliver()
        elif action == 5:  # INTERACT
            reward += self._try_interact()

        # Reward shaping
        reward += self._compute_shaping_reward(prev_pos, self.agent_pos)

        # Inject recharge task if battery critical
        self._check_battery_critical()

        # Increment step
        self.step_count += 1

        # Check termination
        terminated, reason = self._check_termination()
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            reason = 'timeout'
            reward += self.config['rewards']['timeout_penalty']

        if terminated and reason == 'all_tasks_done':
            reward += self.config['rewards']['all_tasks_bonus']

        obs  = self._get_obs()
        info = self._get_info(termination_reason=reason)

        return obs, reward, terminated, truncated, info

    def _try_move_forward(self) -> float:
        """Attempt to move forward. Returns reward delta."""
        dr, dc = DIR_TO_DELTA[self.agent_dir]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        if not self._is_valid_pos(new_pos):
            self.collision_count += 1
            return self.config['rewards']['collision']

        cell = self.grid.get(*new_pos)
        if cell is not None and not cell.can_overlap():
            self.collision_count += 1
            return self.config['rewards']['collision']

        self.agent_pos = new_pos
        return 0.0

    def _try_pickup(self) -> float:
        """Attempt to pick up an item for the current task."""
        if self.carrying_item is not None:
            return 0.0  # Already carrying something
        if self.current_task is None:
            return 0.0
        if self.current_task.pickup_pos is None:
            return 0.0  # Task has no pickup phase
        if self.current_task.pickup_done:
            return 0.0  # Already picked up

        if self._is_adjacent(self.agent_pos, self.current_task.pickup_pos):
            if self.current_task.task_type == TaskType.COFFEE_DELIVERY:
                self.carrying_item = 'coffee'
            elif self.current_task.task_type == TaskType.DOCUMENT_DELIVERY:
                self.carrying_item = 'document'
            self.current_task.pickup_done = True
            return self.config['rewards']['pickup_success']

        return 0.0

    def _try_deliver(self) -> float:
        """Attempt to deliver carried item to current task target."""
        if self.current_task is None:
            return 0.0
        if self.carrying_item is None:
            return 0.0
        if not self.current_task.pickup_done:
            return 0.0

        if self._is_adjacent(self.agent_pos, self.current_task.target_pos):
            self.current_task.completed = True
            self.completed_tasks.append(self.current_task)
            self.carrying_item = None
            reward = self.config['rewards']['task_complete_delivery']
            self._advance_task()
            return reward

        return 0.0

    def _try_interact(self) -> float:
        """Interact with the current cell (meeting room check-in, charging station for task completion)."""
        reward = 0.0

        # Charging station: check if BATTERY_RECHARGE task is complete
        if self.agent_pos == self.charging_pos:
            # If we have a battery recharge task and battery is sufficiently charged, complete it
            if (self.current_task and
                    self.current_task.task_type == TaskType.BATTERY_RECHARGE and
                    self.battery >= 0.8):
                self.current_task.completed = True
                self.completed_tasks.append(self.current_task)
                reward += self.config['rewards']['task_complete_delivery']  # Same reward as delivery
                self._advance_task()

        # Meeting room interaction
        elif (self.current_task and
              self.current_task.task_type == TaskType.MEETING_VISIT and
              self.agent_pos == self.meeting_room_pos):
            self.current_task.completed = True
            self.completed_tasks.append(self.current_task)
            reward += self.config['rewards']['task_complete_meeting']
            self._advance_task()

        return reward

    def _advance_task(self):
        """Move to next task in queue."""
        if self.task_queue:
            self.current_task = self.task_queue.pop(0)
        else:
            self.current_task = None

    def _update_battery(self, action: int) -> float:
        """Update battery level based on action. Returns battery-related reward."""
        reward = 0.0

        # Drain battery first
        self.battery -= self.config['battery_drain_per_step']
        if action == 2:  # MOVE_FORWARD
            self.battery -= self.config['battery_drain_per_move']

        # Recharge if INTERACT at charging station
        if action == 5 and self.agent_pos == self.charging_pos:
            old_battery = self.battery
            self.battery = min(1.0, self.battery + self.config['charge_rate'])
            gained = self.battery - old_battery
            # Reward proportional to battery gained
            reward += gained * self.config['rewards']['battery_efficiency_coef']
            # Bonus if battery is now full or near-full
            if self.battery >= 0.95:
                reward += self.config['rewards']['battery_full_bonus']

        self.battery = max(0.0, self.battery)
        return reward

    def _check_battery_critical(self):
        """Inject recharge task if battery is critically low."""
        critical_threshold = self.config['battery_critical_threshold']
        if self.battery < critical_threshold:
            # Check if recharge task already in queue or current
            has_recharge = False
            if self.current_task and self.current_task.task_type == TaskType.BATTERY_RECHARGE:
                has_recharge = True
            for task in self.task_queue:
                if task.task_type == TaskType.BATTERY_RECHARGE:
                    has_recharge = True
                    break

            if not has_recharge:
                # Create recharge task
                recharge_task = Task(
                    task_id=len(self.all_tasks),
                    task_type=TaskType.BATTERY_RECHARGE,
                    pickup_pos=None,
                    target_pos=self.charging_pos,
                    description="Recharge battery at charging station",
                )
                # If there's a current task, push it back to the queue
                if self.current_task is not None:
                    self.task_queue.insert(0, self.current_task)
                # Make recharge task the current task
                self.current_task = recharge_task
                self.all_tasks.append(recharge_task)

    def _compute_shaping_reward(self, prev_pos: tuple, current_pos: tuple) -> float:
        """Compute distance-based reward shaping."""
        if self.current_task is None:
            return 0.0

        target = self._get_current_subtarget()
        if target is None:
            return 0.0

        prev_dist = abs(prev_pos[0] - target[0]) + abs(prev_pos[1] - target[1])
        curr_dist = abs(current_pos[0] - target[0]) + abs(current_pos[1] - target[1])

        if curr_dist < prev_dist:
            return self.config['rewards']['move_closer']
        elif curr_dist > prev_dist:
            return self.config['rewards']['move_away']
        return 0.0

    def _get_current_subtarget(self) -> tuple:
        """Get the current subtarget (pickup if not done, else delivery target)."""
        if self.current_task is None:
            return None

        # For 2-stage tasks, target is pickup first, then delivery
        if self.current_task.task_type in (TaskType.COFFEE_DELIVERY, TaskType.DOCUMENT_DELIVERY):
            if not self.current_task.pickup_done:
                return self.current_task.pickup_pos
            else:
                return self.current_task.target_pos
        else:
            # Single-stage tasks
            return self.current_task.target_pos

    def _check_termination(self) -> tuple:
        """Check termination conditions. Returns (terminated, reason)."""
        # Condition 1: All tasks completed
        if len(self.task_queue) == 0 and self.current_task is None:
            return True, 'all_tasks_done'

        # Condition 2: Battery depleted
        if self.battery <= 0.0:
            return True, 'battery_dead'

        return False, None

    def _build_grid(self) -> Grid:
        """Construct the 15×15 office grid."""
        grid = Grid(self.grid_size, self.grid_size)

        # Outer walls (MiniGrid convention: grid edges are always walls)
        grid.wall_rect(0, 0, self.grid_size, self.grid_size)

        # Inner walls (top section divider, row 5)
        for col in range(1, 14):
            if col not in (2, 6):  # corridor openings
                grid.set(col, 5, Wall())

        # Inner walls (bottom section divider, row 9)
        for col in range(1, 14):
            if col not in (2, 6, 12):  # corridor openings
                grid.set(col, 9, Wall())

        # Vertical wall separating top offices
        for row in range(1, 5):
            grid.set(4, row, Wall())
        for row in range(1, 5):
            grid.set(8, row, Wall())

        # Place fixed objects
        grid.set(*COFFEE_MACHINE_POS, CoffeeMachine())
        grid.set(*PRINTER_POS, Printer())
        grid.set(*CHARGING_STATION_POS, ChargingStation())
        for i, pos in enumerate(DESK_POSITIONS):
            grid.set(*pos, Desk(i))
        grid.set(*MEETING_TABLE_POS, MeetingTable())

        return grid

    def _get_obs(self) -> np.ndarray:
        """Generate the 53-element observation vector."""
        obs = np.zeros(53, dtype=np.float32)

        # [0-1]   Agent position (normalised by grid size)
        obs[0] = self.agent_pos[0] / self.grid_size
        obs[1] = self.agent_pos[1] / self.grid_size

        # [2]     Agent direction (normalised 0-3 → 0-1)
        obs[2] = self.agent_dir / 3.0

        # [3]     Battery level (already 0.0-1.0)
        obs[3] = self.battery

        # [4]     Carrying item: 0=nothing, 0.5=coffee, 1.0=document
        carrying_map = {None: 0.0, 'coffee': 0.5, 'document': 1.0}
        obs[4] = carrying_map.get(self.carrying_item, 0.0)

        # [5-9]   Task type one-hot (none, coffee, document, meeting, charge)
        task_type_idx = self._get_task_type_idx()
        if task_type_idx >= 0:
            obs[5 + task_type_idx] = 1.0

        # [10-11] Current task target position (normalised), if any
        if self.current_task and self.current_task.target_pos:
            obs[10] = self.current_task.target_pos[0] / self.grid_size
            obs[11] = self.current_task.target_pos[1] / self.grid_size

        # [12-13] Pickup location for current task (normalised), if any
        if self.current_task and self.current_task.pickup_pos:
            obs[12] = self.current_task.pickup_pos[0] / self.grid_size
            obs[13] = self.current_task.pickup_pos[1] / self.grid_size

        # [14]    Number of remaining tasks (normalised by max tasks)
        obs[14] = len(self.task_queue) / self.max_tasks

        # [15-34] 5×4 local view: type encoding of 5 cells ahead/around agent
        local_cells = self._get_local_view(radius=2)
        obs[15:35] = local_cells

        # [35-52] Key location encodings (normalised)
        #   coffee_machine_pos, printer_pos, charging_pos, desk1, desk2, desk3
        fixed_positions = [
            self.coffee_machine_pos,   # [35-36]
            self.printer_pos,          # [37-38]
            self.charging_pos,         # [39-40]
            self.desk_positions[0],    # [41-42]
            self.desk_positions[1],    # [43-44]
            self.desk_positions[2],    # [45-46]
        ]
        for i, pos in enumerate(fixed_positions):
            obs[35 + i*2]     = pos[0] / self.grid_size
            obs[35 + i*2 + 1] = pos[1] / self.grid_size

        # [47-52] Available for future extensions (zero-padded)

        return obs

    def _get_task_type_idx(self) -> int:
        """Get task type index for one-hot encoding."""
        if self.current_task is None:
            return -1
        task_type = self.current_task.task_type
        if task_type == TaskType.COFFEE_DELIVERY:
            return 0
        elif task_type == TaskType.DOCUMENT_DELIVERY:
            return 1
        elif task_type == TaskType.MEETING_VISIT:
            return 2
        elif task_type == TaskType.BATTERY_RECHARGE:
            return 3
        return -1

    def _get_local_view(self, radius: int = 2) -> np.ndarray:
        """Returns a flattened float array of cells around the agent."""
        view = []
        # Sample 5 cells: forward-1, forward-2, left, right, behind (relative to agent)
        offsets = self._direction_offsets(radius)
        for r, c in offsets:
            cell = self.grid.get(r, c) if self._is_valid_pos((r, c)) else None
            if cell is None:
                view.extend([0.0, 0.0, 0.0, 0.0])
            elif isinstance(cell, Wall):
                view.extend([0.0, 1.0, 0.0, 0.0])
            elif isinstance(cell, (CoffeeMachine, Printer, ChargingStation, Desk, MeetingTable)):
                view.extend([0.0, 0.0, 1.0, 0.0])
            else:
                view.extend([0.0, 0.0, 0.0, 1.0])
        return np.array(view[:20], dtype=np.float32)

    def _direction_offsets(self, radius: int) -> list:
        """Get 5 cell positions relative to agent's current position and direction."""
        dr, dc = DIR_TO_DELTA[self.agent_dir]
        r, c = self.agent_pos

        # Forward cells
        forward_cells = [(r + dr*i, c + dc*i) for i in range(1, radius + 1)]

        # Side cells (perpendicular to forward direction)
        if self.agent_dir == 0 or self.agent_dir == 2:  # facing left/right
            left = (r - 1, c)
            right = (r + 1, c)
        else:  # facing up/down
            left = (r, c - 1)
            right = (r, c + 1)

        # Behind cell
        behind = (r - dr, c - dc)

        # Return 5 cells: forward-1, forward-2, left, right, behind
        result = forward_cells + [left, right, behind]
        return result[:5]

    def _get_info(self, termination_reason: str = None) -> dict:
        """Generate the info dictionary."""
        info = {
            'battery': self.battery,
            'carrying_item': self.carrying_item,
            'current_task': str(self.current_task.description) if self.current_task else None,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len([t for t in self.all_tasks if t.failed]),
            'collision_count': self.collision_count,
            'step_count': self.step_count,
            'termination_reason': termination_reason,
            'battery_low': self.battery < self.config['battery_low_threshold'],
            'task_queue_length': len(self.task_queue),
        }
        return info

    def _is_valid_pos(self, pos: tuple) -> bool:
        """Check if position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _is_adjacent(self, pos_a: tuple, pos_b: tuple) -> bool:
        """Returns True if pos_a is adjacent (4-connected) to pos_b."""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) <= 1

    def render(self):
        """Render the environment using MiniGrid's Grid.render().

        Returns:
            np.ndarray (H, W, 3) if render_mode=='rgb_array', else None.
        """
        if self.grid is None:
            return None

        tile_size = self.config.get('tile_size', 32)

        # MiniGrid Grid.render expects (col, row) for agent_pos
        agent_pos_colrow = (self.agent_pos[1], self.agent_pos[0])

        img = self.grid.render(
            tile_size=tile_size,
            agent_pos=agent_pos_colrow,
            agent_dir=self.agent_dir,
        )

        if self.render_mode == 'rgb_array':
            return img

        if self.render_mode == 'human':
            import pygame
            if self.window is None:
                pygame.init()
                pygame.display.init()
                h, w = img.shape[:2]
                self.window = pygame.display.set_mode((w, h))
                pygame.display.set_caption('OfficeBot RL')
                self.clock = pygame.time.Clock()

            # Pump events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            # img is (H, W, 3) — pygame surface wants (W, H)
            surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
            self.window.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata.get('render_fps', 10))

        return None

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
