# realistic_solar_system.py

from solar_system import SolarSystem, Sun, Planet
from solar_constants import SOLAR_SYSTEM_MASSES, SOLAR_SYSTEM_POS, SOLAR_SYSTEM_VEL, SOLAR_SYSTEM_COLORS

# Create solar system (bigger size for outer planets)
solar_system = SolarSystem(60, projection_2d=False)  # 60 AU size

# Create Sun with real data
sun = Sun(
    solar_system,
    mass=SOLAR_SYSTEM_MASSES["Sun"],
    position=tuple(SOLAR_SYSTEM_POS["Sun"]),
    velocity=tuple(SOLAR_SYSTEM_VEL["Sun"])
)

# Create Earth with real data
earth = Planet(
    solar_system,
    mass=SOLAR_SYSTEM_MASSES["Earth"],
    position=tuple(SOLAR_SYSTEM_POS["Earth"]),
    velocity=tuple(SOLAR_SYSTEM_VEL["Earth"])
)
earth.colour = SOLAR_SYSTEM_COLORS["Earth"]

# Create Mars with real data
mars = Planet(
    solar_system,
    mass=SOLAR_SYSTEM_MASSES["Mars"],
    position=tuple(SOLAR_SYSTEM_POS["Mars"]),
    velocity=tuple(SOLAR_SYSTEM_VEL["Mars"])
)
mars.colour = SOLAR_SYSTEM_COLORS["Mars"]

# Add more planets if desired (Venus, Jupiter, etc.)

# Run simulation
while True:
    solar_system.update_all()
    solar_system.draw_all()
