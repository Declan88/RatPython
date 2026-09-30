"""
A player's inventory: the weapons they carry and which one is out. Any weapon in
the registry (see registry.py) can go in it, by id.

    inventory = Inventory(["usp", "gouda_gun"])
    inventory.add("some_new_gun")
    inventory.select(1)  /  inventory.step(+1)   # scroll wheel

It only tracks the weapons and the selection - loading, showing and hiding them
is the game's job (app.py's switch_weapon), driven by the `on_added`/`on_removed`
callbacks.
"""

from .registry import create_weapon, get_weapon_class


class Inventory:
    def __init__(self, loadout=(), capacity=None):
        """loadout: weapon ids to start with. capacity: the most weapons it can
        hold (None = unlimited)."""
        self.weapons = []          # WeaponsBase instances, one per slot
        self.index = 0             # the selected slot
        self.capacity = capacity
        self.on_added = None       # callback(weapon) after a weapon is added
        self.on_removed = None     # callback(weapon) after one is removed
        for weapon_id in loadout:
            self.add(weapon_id)

    @property
    def current(self):
        return self.weapons[self.index] if self.weapons else None

    def __len__(self):
        return len(self.weapons)

    def has(self, weapon_id):
        return any(type(w) is get_weapon_class(weapon_id) for w in self.weapons)

    def add(self, weapon_id):
        """Puts a new weapon in the next slot. Returns it, or None if the id is
        unknown, it's already carried, or the inventory is full."""
        cls = get_weapon_class(weapon_id)
        if cls is None or self.has(weapon_id):
            return None
        if self.capacity is not None and len(self.weapons) >= self.capacity:
            return None
        weapon = cls()
        self.weapons.append(weapon)
        if self.on_added is not None:
            self.on_added(weapon)
        return weapon

    def remove(self, weapon_id):
        """Takes a weapon out. The selection stays on the same weapon where it can."""
        cls = get_weapon_class(weapon_id)
        for i, weapon in enumerate(self.weapons):
            if type(weapon) is cls:
                current = self.current
                self.weapons.pop(i)
                self.index = self.weapons.index(current) if current in self.weapons else min(
                    self.index, max(0, len(self.weapons) - 1))
                if self.on_removed is not None:
                    self.on_removed(weapon)
                return weapon
        return None

    def select(self, index):
        """Selects a slot (wrapping). Returns the weapon there, or None if empty."""
        if not self.weapons:
            return None
        self.index = index % len(self.weapons)
        return self.current

    def step(self, direction):
        """Selects the next (+1) or previous (-1) slot, wrapping."""
        return self.select(self.index + direction)
