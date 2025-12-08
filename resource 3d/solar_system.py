# solar_system_3d.py

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
from vectors import Vector

class SolarSystem:
    def __init__(self, size, projection_2d=False):
        self.size = size
        self.projection_2d = projection_2d
        self.bodies = []

        self.fig, self.ax = plt.subplots(
            1,
            1,
            subplot_kw={"projection": "3d"},
            figsize=(self.size / 50, self.size / 50),
        )
        if self.projection_2d:
            self.ax.view_init(10, 0)
        else:
            self.ax.view_init(0, 0)
        self.fig.tight_layout()

    def add_body(self, body):
        self.bodies.append(body)

    def update_all(self):
        # Calculate accelerations for all bodies using 5 Steps physics
        accel = self.calculate_accelerations()
        dt = 10  # 1 day timestep (matches 5 Steps units)
    
        # Update velocities and positions using Euler integration
        for i, body in enumerate(self.bodies):
            # Convert np acceleration to Vector before adding
            a_vec = Vector(*accel[i])
            body.velocity += a_vec * dt

            body.position = (
                body.position[0] + body.velocity.x * dt,
                body.position[1] + body.velocity.y * dt,
                body.position[2] + body.velocity.z * dt,
            )

        # Sort bodies for drawing (keep this if you need it)
        self.bodies.sort(key=lambda item: item.position[0])

        # Draw all bodies
        for body in self.bodies:
            body.draw()


    def draw_all(self):
        self.ax.set_xlim((-self.size / 2, self.size / 2))
        self.ax.set_ylim((-self.size / 2, self.size / 2))
        self.ax.set_zlim((-self.size / 2, self.size / 2))
        if self.projection_2d:
            self.ax.xaxis.set_ticklabels([])
            self.ax.yaxis.set_ticklabels([])
            self.ax.zaxis.set_ticklabels([])
        else:
            self.ax.axis(False)
        plt.pause(0.001)
        self.ax.clear()

    def calculate_accelerations(self):
        from solar_constants import G  # Import the real G value
    
        N = len(self.bodies)
        pos = np.array([b.position for b in self.bodies])    # shape (N, 3)
        mass = np.array([b.mass for b in self.bodies])
        accel = np.zeros((N, 3))

        r_ij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r_norm = np.linalg.norm(r_ij, axis=2)
        with np.errstate(divide="ignore"):
            inv_r_cubed = 1.0 / (r_norm * r_norm * r_norm)
        np.fill_diagonal(inv_r_cubed, 0.0)
        for i in range(N):
            # Vector sum over all others (from 5 Steps)
            accel[i] = G * np.sum(r_ij[i] * inv_r_cubed[i][:, None] * mass[:, None], axis=0)
        return accel


class SolarSystemBody:
    min_display_size = 10
    display_log_base = 1.3

    def __init__(
        self,
        solar_system,
        mass,
        position=(0, 0, 0),
        velocity=(0, 0, 0),
    ):
        self.solar_system = solar_system
        self.mass = mass
        self.position = position
        self.velocity = Vector(*velocity)
        self.display_size = max(
            math.log(self.mass, self.display_log_base),
            self.min_display_size,
        )
        self.colour = "black"

        self.solar_system.add_body(self)

    def move(self):
        self.position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1],
            self.position[2] + self.velocity[2],
        )

    def draw(self):
        self.solar_system.ax.plot(
            *self.position,
            marker="o",
            markersize=self.display_size + self.position[0] / 30,
            color=self.colour
        )
        if self.solar_system.projection_2d:
            self.solar_system.ax.plot(
                self.position[0],
                self.position[1],
                -self.solar_system.size / 2,
                marker="o",
                markersize=self.display_size / 2,
                color=(.5, .5, .5),
            )



class Sun(SolarSystemBody):
    def __init__(
        self,
        solar_system,
        mass=10_000,
        position=(0, 0, 0),
        velocity=(0, 0, 0),
    ):
        super(Sun, self).__init__(solar_system, mass, position, velocity)
        self.colour = "yellow"

class Planet(SolarSystemBody):
    colours = itertools.cycle([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

    def __init__(
        self,
        solar_system,
        mass=10,
        position=(0, 0, 0),
        velocity=(0, 0, 0),
    ):
        super(Planet, self).__init__(solar_system, mass, position, velocity)
        self.colour = next(Planet.colours)