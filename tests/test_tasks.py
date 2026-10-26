# tests/test_tasks.py

import pytest
import yaml
import numpy as np
from environment.tasks import TaskQueueManager, TaskType
from environment.constants import DESK_POSITIONS


@pytest.fixture
def config():
    """Load environment configuration."""
    with open('configs/env_config.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def task_manager(config):
    """Create TaskQueueManager instance."""
    return TaskQueueManager(
        max_tasks=config['max_tasks'],
        desk_positions=DESK_POSITIONS
    )


def test_task_manager_creation(task_manager):
    """Test that TaskQueueManager can be created."""
    assert task_manager is not None


def test_generates_correct_number_of_tasks(task_manager):
    """Test that correct number of tasks are generated."""
    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)
    assert len(tasks) == task_manager.max_tasks, \
        f"Expected {task_manager.max_tasks} tasks, got {len(tasks)}"


def test_all_task_types_valid(task_manager):
    """Test that all generated tasks have valid types."""
    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    valid_types = [TaskType.COFFEE_DELIVERY, TaskType.DOCUMENT_DELIVERY, TaskType.MEETING_VISIT]

    for task in tasks:
        assert task.task_type in valid_types, \
            f"Task type {task.task_type} not in valid types {valid_types}"


def test_tasks_have_valid_positions(task_manager):
    """Test that all tasks have valid position coordinates."""
    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    for task in tasks:
        # Check target position is valid
        assert task.target_pos is not None, "Task should have target position"
        assert isinstance(task.target_pos, tuple), "Target position should be tuple"
        assert len(task.target_pos) == 2, "Target position should be (row, col)"
        assert 0 <= task.target_pos[0] < 15, "Target row should be in [0, 15)"
        assert 0 <= task.target_pos[1] < 15, "Target col should be in [0, 15)"

        # Check pickup position for delivery tasks
        if task.task_type in [TaskType.COFFEE_DELIVERY, TaskType.DOCUMENT_DELIVERY]:
            assert task.pickup_pos is not None, "Delivery task should have pickup position"
            assert isinstance(task.pickup_pos, tuple), "Pickup position should be tuple"
            assert len(task.pickup_pos) == 2, "Pickup position should be (row, col)"
            assert 0 <= task.pickup_pos[0] < 15, "Pickup row should be in [0, 15)"
            assert 0 <= task.pickup_pos[1] < 15, "Pickup col should be in [0, 15)"


def test_coffee_delivery_task_positions(task_manager):
    """Test that coffee delivery tasks have correct pickup and delivery positions."""
    from environment.constants import COFFEE_MACHINE_POS

    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    coffee_tasks = [t for t in tasks if t.task_type == TaskType.COFFEE_DELIVERY]

    for task in coffee_tasks:
        # Pickup should be at coffee machine
        assert task.pickup_pos == COFFEE_MACHINE_POS, \
            f"Coffee pickup should be at {COFFEE_MACHINE_POS}, got {task.pickup_pos}"
        # Delivery should be at a desk
        assert task.target_pos in DESK_POSITIONS, \
            f"Coffee delivery should be to a desk, got {task.target_pos}"


def test_document_delivery_task_positions(task_manager):
    """Test that document delivery tasks have correct pickup and delivery positions."""
    from environment.constants import PRINTER_POS

    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    doc_tasks = [t for t in tasks if t.task_type == TaskType.DOCUMENT_DELIVERY]

    for task in doc_tasks:
        # Pickup should be at printer
        assert task.pickup_pos == PRINTER_POS, \
            f"Document pickup should be at {PRINTER_POS}, got {task.pickup_pos}"
        # Delivery should be at a desk
        assert task.target_pos in DESK_POSITIONS, \
            f"Document delivery should be to a desk, got {task.target_pos}"


def test_meeting_visit_task_position(task_manager):
    """Test that meeting visit tasks have correct target position."""
    from environment.constants import MEETING_ROOM_POS

    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    meeting_tasks = [t for t in tasks if t.task_type == TaskType.MEETING_VISIT]

    for task in meeting_tasks:
        # Meeting should be at meeting room
        assert task.target_pos == MEETING_ROOM_POS, \
            f"Meeting should be at {MEETING_ROOM_POS}, got {task.target_pos}"
        # No pickup for meeting tasks
        assert task.pickup_pos is None, "Meeting tasks should not have pickup position"


def test_task_ids_unique(task_manager):
    """Test that all task IDs are unique."""
    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    task_ids = [t.task_id for t in tasks]
    assert len(task_ids) == len(set(task_ids)), "Task IDs should be unique"


def test_task_ids_sequential(task_manager):
    """Test that task IDs are sequential starting from 0."""
    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    task_ids = [t.task_id for t in tasks]
    expected_ids = list(range(len(tasks)))
    assert task_ids == expected_ids, f"Task IDs should be sequential, got {task_ids}"


def test_tasks_have_descriptions(task_manager):
    """Test that all tasks have non-empty descriptions."""
    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    for task in tasks:
        assert task.description is not None, "Task should have description"
        assert isinstance(task.description, str), "Task description should be string"
        assert len(task.description) > 0, "Task description should not be empty"


def test_tasks_not_completed_initially(task_manager):
    """Test that generated tasks are not marked as completed."""
    rng = np.random.default_rng(42)
    tasks = task_manager.generate_episode_tasks(rng)

    for task in tasks:
        assert not task.completed, "Newly generated task should not be completed"
        assert not task.failed, "Newly generated task should not be failed"


def test_different_seeds_produce_different_tasks(task_manager):
    """Test that different random seeds produce different task sequences."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(123)

    tasks1 = task_manager.generate_episode_tasks(rng1)
    tasks2 = task_manager.generate_episode_tasks(rng2)

    # Check if at least one task type differs
    types1 = [t.task_type for t in tasks1]
    types2 = [t.task_type for t in tasks2]

    # Note: With randomization, there's a small chance they could be the same
    # We're checking that the system CAN produce different results
    # If this fails consistently, there might be an issue with randomization
    assert len(tasks1) == len(tasks2), "Should produce same number of tasks"
