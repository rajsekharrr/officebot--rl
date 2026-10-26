# environment/tasks.py

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, List
import numpy as np

from environment.constants import (
    COFFEE_MACHINE_POS,
    PRINTER_POS,
    CHARGING_STATION_POS,
    MEETING_ROOM_POS,
)


class TaskType(IntEnum):
    COFFEE_DELIVERY  = 0
    DOCUMENT_DELIVERY = 1
    MEETING_VISIT    = 2
    BATTERY_RECHARGE = 3


@dataclass
class Task:
    task_id: int
    task_type: TaskType
    pickup_pos: Optional[Tuple[int, int]]   # Where to pick up item
    target_pos: Tuple[int, int]             # Where to deliver/visit
    description: str
    completed: bool = False
    failed: bool = False
    steps_taken: int = 0
    pickup_done: bool = False               # For 2-stage tasks


TASK_TEMPLATES = {
    TaskType.COFFEE_DELIVERY: {
        'pickup_pos': COFFEE_MACHINE_POS,   # from env layout constants
        'target_pos': None,                 # randomly assigned desk at episode start
        'description': "Deliver coffee to Desk {desk_id}",
    },
    TaskType.DOCUMENT_DELIVERY: {
        'pickup_pos': PRINTER_POS,
        'target_pos': None,
        'description': "Deliver document to Desk {desk_id}",
    },
    TaskType.MEETING_VISIT: {
        'pickup_pos': None,                 # no pickup needed
        'target_pos': MEETING_ROOM_POS,
        'description': "Attend meeting in meeting room",
    },
    TaskType.BATTERY_RECHARGE: {
        'pickup_pos': None,
        'target_pos': CHARGING_STATION_POS,
        'description': "Recharge battery at charging station",
    },
}


class TaskQueueManager:
    """Manages the agent's task queue for one episode."""

    def __init__(self, max_tasks: int = 5, desk_positions: list = None):
        self.max_tasks = max_tasks
        self.desk_positions = desk_positions  # list of (row, col)

    def generate_episode_tasks(self, rng: np.random.Generator) -> List[Task]:
        """Generate a randomised but sensible set of tasks for one episode."""
        tasks = []
        task_types = [
            TaskType.COFFEE_DELIVERY,
            TaskType.DOCUMENT_DELIVERY,
            TaskType.MEETING_VISIT,
        ]
        selected_types = rng.choice(task_types, size=self.max_tasks, replace=True)

        for i, t_type in enumerate(selected_types):
            desk_idx = rng.integers(0, len(self.desk_positions))
            desk_pos = self.desk_positions[desk_idx]
            template = TASK_TEMPLATES[t_type].copy()
            if t_type in (TaskType.COFFEE_DELIVERY, TaskType.DOCUMENT_DELIVERY):
                template['target_pos'] = desk_pos
                template['description'] = template['description'].format(
                    desk_id=desk_idx + 1)
            tasks.append(Task(
                task_id=i,
                task_type=t_type,
                pickup_pos=template['pickup_pos'],
                target_pos=template['target_pos'],
                description=template['description'],
            ))
        return tasks
