# environment/constants.py
# All fixed grid positions. Row first, then column.

GRID_SIZE = 15

COFFEE_MACHINE_POS    = (2, 6)
PRINTER_POS           = (7, 2)
CHARGING_STATION_POS  = (11, 2)
MEETING_ROOM_POS      = (3, 11)   # Check-in point inside meeting room
MEETING_TABLE_POS     = (11, 12)

DESK_POSITIONS = [
    (2, 2),    # Desk 1 — top-left office
    (7, 10),   # Desk 2 — middle-right office
    (11, 10),  # Desk 3 — bottom-right office
]

AGENT_START_POS = (13, 4)

# Direction vectors: 0=right, 1=down, 2=left, 3=up (MiniGrid convention)
DIR_TO_DELTA = {
    0: (0,  1),   # right → increase col
    1: (1,  0),   # down  → increase row
    2: (0, -1),   # left  → decrease col
    3: (-1, 0),   # up    → decrease row
}
