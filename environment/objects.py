# environment/objects.py

import numpy as np
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

# Colour palette (RGB) for each object type
_YELLOW  = np.array([255, 220,   0])   # CoffeeMachine
_BLUE    = np.array([  0, 128, 255])   # Printer
_GREEN   = np.array([  0, 200,  80])   # ChargingStation
_GREY    = np.array([160, 160, 160])   # Desk
_PURPLE  = np.array([160,  32, 240])   # MeetingTable


class CoffeeMachine(WorldObj):
    """Pickup point for coffee delivery tasks — rendered as a yellow circle."""

    def __init__(self):
        super().__init__('ball', 'yellow')
        self.obj_type = 'coffee_machine'

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def encode(self):
        return (10, 0, 0)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), _YELLOW)


class Printer(WorldObj):
    """Pickup point for document delivery tasks — rendered as a blue square."""

    def __init__(self):
        super().__init__('ball', 'blue')
        self.obj_type = 'printer'

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def encode(self):
        return (11, 0, 0)

    def render(self, img):
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), _BLUE)


class ChargingStation(WorldObj):
    """Agent recharges battery here — rendered as a green circle."""

    def __init__(self):
        super().__init__('ball', 'green')
        self.obj_type = 'charging_station'

    def can_overlap(self):
        return True   # Agent can stand on it

    def encode(self):
        return (12, 0, 0)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), _GREEN)
        # Small inner dot to distinguish from desks
        fill_coords(img, point_in_circle(0.5, 0.5, 0.12), np.array([255, 255, 255]))


class Desk(WorldObj):
    """Delivery destination — rendered as a grey rectangle."""

    def __init__(self, desk_id: int):
        super().__init__('ball', 'grey')
        self.obj_type = 'desk'
        self.desk_id = desk_id

    def can_overlap(self):
        return False

    def encode(self):
        return (13, self.desk_id, 0)

    def render(self, img):
        fill_coords(img, point_in_rect(0.15, 0.85, 0.35, 0.65), _GREY)
        fill_coords(img, point_in_rect(0.35, 0.65, 0.25, 0.75), _GREY)


class MeetingTable(WorldObj):
    """Visual decoration inside meeting room — rendered as a purple oval."""

    def __init__(self):
        super().__init__('ball', 'purple')
        self.obj_type = 'meeting_table'

    def can_overlap(self):
        return False

    def encode(self):
        return (14, 0, 0)

    def render(self, img):
        fill_coords(img, point_in_rect(0.15, 0.85, 0.25, 0.75), _PURPLE)
