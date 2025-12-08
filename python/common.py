from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

##### Step 1 #####
class System:
    def __init__(
        self, num_particles: int, x: np.ndarray, v: np.ndarray, m: np.ndarray, G: float
    ) -> None:
        self.num_particles = num_particles
        self.x = x
        self.v = v
        self.m = m
        self.G = G

    def center_of_mass_correction(self) -> None:
        """Set center of mass of position and velocity to zero"""
        M = np.sum(self.m)
        x_cm = np.einsum("i,ij->j", self.m, self.x) / M
        v_cm = np.einsum("i,ij->j", self.m, self.v) / M
        self.x -= x_cm
        self.v -= v_cm


def get_initial_conditions(
    initial_condition: str,
) -> Tuple[System, List[Optional[str]], List[Optional[str]], bool]:
    """
    Returns the initial conditions for solar system,
    with units AU, days, and M_sun-equivalent (m are μ/μ☉ so G*m = μ).
    """
    # Conversion factor from km^3 s^-2 to AU^3 d^-2
    CONVERSION_FACTOR = (86400**2) / (149597870.7**3)

    # GM values (km^3 s^-2)
    # ref: Park et al. 2021 (DE440/441)
    GM_KM_S = {
        "Sun": 132712440041.279419,
        "Mercury": 22031.868551,
        "Venus": 324858.592000,
        "Earth": 398600.435507,
        "Mars": 42828.375816,
        "Jupiter": 126712764.100000,
        "Saturn": 37940584.841800,
        "Uranus": 5794556.400000,
        "Neptune": 6836527.100580,
        "Moon": 4902.800118,
        "Pluto": 975.500000,
        "Ceres": 62.62890,
        "Vesta": 17.288245,
    }

    # GM values (AU^3 d^-2)
    GM_AU_DAY = {k: v * CONVERSION_FACTOR for k, v in GM_KM_S.items()}

    # Store masses as μ/μ☉ so that G * m_j = μ_j in AU^3 d^-2
    SOLAR_SYSTEM_MASSES = {k: GM_KM_S[k] / GM_KM_S["Sun"] for k in GM_KM_S}

    # Use μ☉ as "G"; with m_j = μ_j/μ☉ we get G*m_j = μ_j (desired)
    G = GM_AU_DAY["Sun"]

    # Solar system position and velocities (AU, AU/day), barycentric, JPL Horizons, 2024-01-01 TDB
    SOLAR_SYSTEM_POS = {
        "Sun": [-7.967955691533730e-03, -2.906227441573178e-03, 2.103054301547123e-04],
        "Mercury": [-2.825983269538632e-01, 1.974559795958082e-01, 4.177433558063677e-02],
        "Venus": [-7.232103701666379e-01, -7.948302026312400e-02, 4.042871428174315e-02],
        "Earth": [-1.738192017257054e-01, 9.663245550235138e-01, 1.553901854897183e-04],
        "Mars": [-3.013262392582653e-01, -1.454029331393295e00, -2.300531433991428e-02],
        "Jupiter": [3.485202469657674e00, 3.552136904413157e00, -9.271035442798399e-02],
        "Saturn": [8.988104223143450e00, -3.719064854634689e00, -2.931937777323593e-01],
        "Uranus": [1.226302417897505e01, 1.529738792480545e01, -1.020549026883563e-01],
        "Neptune": [2.983501460984741e01, -1.793812957956852e00, -6.506401132254588e-01],
        "Moon": [-1.762788124769829e-01, 9.674377513177153e-01, 3.236901585768862e-04],
        "Pluto": [1.720200478843485e01, -3.034155683573043e01, -1.729127607100611e00],
        "Ceres": [-1.103880510367569e00, -2.533340440444230e00, 1.220283937721780e-01],
        "Vesta": [-8.092549658731499e-02, 2.558381434460076e00, -6.695836142398572e-02],
    }
    SOLAR_SYSTEM_VEL = {
        "Sun": [4.875094764261564e-06, -7.057133213976680e-06, -4.573453713094512e-08],
        "Mercury": [-2.232165900189702e-02, -2.157207103176252e-02, 2.855193410495743e-04],
        "Venus": [2.034068201002341e-03, -2.020828626592994e-02, -3.945639843855159e-04],
        "Earth": [-1.723001232538228e-02, -2.967721342618870e-03, 6.382125383116755e-07],
        "Mars": [1.424832259345280e-02, -1.579236181580905e-03, -3.823722796161561e-04],
        "Jupiter": [-5.470970658852281e-03, 5.642487338479145e-03, 9.896190602066252e-05],
        "Saturn": [1.822013845554067e-03, 5.143470425888054e-03, -1.617235904887937e-04],
        "Uranus": [-3.097615358317413e-03, 2.276781932345769e-03, 4.860433222241686e-05],
        "Neptune": [1.676536611817232e-04, 3.152098732861913e-03, -6.877501095688201e-05],
        "Moon": [-1.746667306153906e-02, -3.473438277358121e-03, -3.359028758606074e-05],
        "Pluto": [2.802810313667557e-03, 8.492056438614633e-04, -9.060790113327894e-04],
        "Ceres": [8.978653480111301e-03, -4.873256528198994e-03, -1.807162046049230e-03],
        "Vesta": [-1.017876585480054e-02, -5.452367109338154e-04, 1.255870551153315e-03],
    }

    SOLAR_SYSTEM_COLORS = {
        "Sun": "orange",
        "Mercury": "slategrey",
        "Venus": "wheat",
        "Earth": "skyblue",
        "Mars": "red",
        "Jupiter": "darkgoldenrod",
        "Saturn": "gold",
        "Uranus": "paleturquoise",
        "Neptune": "blue",
    }

    SOLAR_SYSTEM_PLUS_COLORS = {
        "Sun": "orange",
        "Mercury": "slategrey",
        "Venus": "wheat",
        "Earth": "skyblue",
        "Mars": "red",
        "Jupiter": "darkgoldenrod",
        "Saturn": "gold",
        "Uranus": "paleturquoise",
        "Neptune": "blue",
        "Pluto": None,
        "Ceres": None,
        "Vesta": None,
    }

    if initial_condition == "pyth-3-body":
        # Pythagorean 3-body problem
        R1 = np.array([1.0, 3.0, 0.0])
        R2 = np.array([-2.0, -1.0, 0.0])
        R3 = np.array([1.0, -1.0, 0.0])
        V1 = np.array([0.0, 0.0, 0.0])
        V2 = np.array([0.0, 0.0, 0.0])
        V3 = np.array([0.0, 0.0, 0.0])

        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
        m = np.array([3.0 / G, 4.0 / G, 5.0 / G])

        system = System(num_particles=len(m), x=x, v=v, m=m, G=G)
        system.center_of_mass_correction()

        labels: List[Optional[str]] = [None, None, None]
        colors: List[Optional[str]] = [None, None, None]
        legend = False

        return system, labels, colors, legend

    elif initial_condition == "solar_system":
        m = np.array(
            [
                SOLAR_SYSTEM_MASSES["Sun"],
                SOLAR_SYSTEM_MASSES["Mercury"],
                SOLAR_SYSTEM_MASSES["Venus"],
                SOLAR_SYSTEM_MASSES["Earth"],
                SOLAR_SYSTEM_MASSES["Mars"],
                SOLAR_SYSTEM_MASSES["Jupiter"],
                SOLAR_SYSTEM_MASSES["Saturn"],
                SOLAR_SYSTEM_MASSES["Uranus"],
                SOLAR_SYSTEM_MASSES["Neptune"],
            ]
        )

        names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"]
        x = np.array([np.array(SOLAR_SYSTEM_POS[n]) for n in names])
        v = np.array([np.array(SOLAR_SYSTEM_VEL[n]) for n in names])

        system = System(num_particles=len(m), x=x, v=v, m=m, G=G)
        system.center_of_mass_correction()

        labels = list(SOLAR_SYSTEM_COLORS.keys())
        colors = list(SOLAR_SYSTEM_COLORS.values())
        legend = True

        return system, labels, colors, legend

    elif initial_condition == "solar_system_plus":
        m = np.array(
            [
                SOLAR_SYSTEM_MASSES["Sun"],
                SOLAR_SYSTEM_MASSES["Mercury"],
                SOLAR_SYSTEM_MASSES["Venus"],
                SOLAR_SYSTEM_MASSES["Earth"],
                SOLAR_SYSTEM_MASSES["Mars"],
                SOLAR_SYSTEM_MASSES["Jupiter"],
                SOLAR_SYSTEM_MASSES["Saturn"],
                SOLAR_SYSTEM_MASSES["Uranus"],
                SOLAR_SYSTEM_MASSES["Neptune"],
                SOLAR_SYSTEM_MASSES["Pluto"],
                SOLAR_SYSTEM_MASSES["Ceres"],
                SOLAR_SYSTEM_MASSES["Vesta"],
            ]
        )

        names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto","Ceres","Vesta"]
        x = np.array([np.array(SOLAR_SYSTEM_POS[n]) for n in names])
        v = np.array([np.array(SOLAR_SYSTEM_VEL[n]) for n in names])

        system = System(num_particles=len(m), x=x, v=v, m=m, G=G)
        system.center_of_mass_correction()

        labels = list(SOLAR_SYSTEM_PLUS_COLORS.keys())
        colors = list(SOLAR_SYSTEM_PLUS_COLORS.values())
        legend = True

        return system, labels, colors, legend

    else:
        raise ValueError(f"Initial condition not recognized: {initial_condition}.")


def plot_initial_conditions(
    system: System,
    labels: list,
    colors: list,
    legend: bool,
) -> None:
    """Quick scatter of initial positions in AU."""
    fig, ax = plt.subplots()
    ax.set_xlabel("$x$ (AU)")
    ax.set_ylabel("$y$ (AU)")
    for i in range(system.num_particles):
        ax.scatter(system.x[i, 0], system.x[i, 1], marker="o", color=colors[i], label=labels[i])
    if legend:
        ax.legend()
    plt.show()


##### Step 2 #####
def acceleration(a: np.ndarray, system: System) -> None:
    """
    Compute gravitational acceleration with tiny Plummer softening.
    """
    a.fill(0.0)
    x = system.x
    m = system.m
    G = system.G

    r_ij = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r2 = np.einsum("ijk,ijk->ij", r_ij, r_ij)
    eps2 = 1e-12  # (1e-6 AU)^2 softening
    r2 += np.eye(len(m)) * 0.0  # keep shape
    inv_r = 1.0 / np.sqrt(r2 + eps2)
    inv_r3 = inv_r / (r2 + eps2)
    np.fill_diagonal(inv_r3, 0.0)

    a[:] = G * np.einsum("ijk,ij,i->jk", r_ij, inv_r3, m)


@jit(nopython=True)
def acceleration_numba(a, x, m, G, num_particles):
    """Numba-optimized gravitational acceleration with tiny softening"""
    a.fill(0.0)
    eps2 = 1e-12  # (1e-6 AU)^2
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                dx = x[j, 0] - x[i, 0]
                dy = x[j, 1] - x[i, 1]
                dz = x[j, 2] - x[i, 2]
                r2 = dx*dx + dy*dy + dz*dz + eps2
                inv_r = 1.0 / np.sqrt(r2)
                inv_r3 = inv_r / r2
                a[i, 0] += G * m[j] * dx * inv_r3
                a[i, 1] += G * m[j] * dy * inv_r3
                a[i, 2] += G * m[j] * dz * inv_r3


##### Step 3 #####
def euler(a: np.ndarray, system: System, dt: float) -> None:
    """Simple Euler step (not symplectic; for reference only)."""
    acceleration(a, system)
    system.x += system.v * dt
    system.v += a * dt


@jit(nopython=True)
def velocity_verlet_numba(a, x, v, m, G, num_particles, dt):
    """Velocity Verlet (symplectic) with tiny softening"""
    # a is acceleration at current positions
    a_old = a.copy()

    # Drift positions
    x[:] = x + v * dt + 0.5 * a_old * dt * dt

    # Recompute accelerations at new positions
    a.fill(0.0)
    eps2 = 1e-12  # (1e-6 AU)^2
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                dx = x[j, 0] - x[i, 0]
                dy = x[j, 1] - x[i, 1]
                dz = x[j, 2] - x[i, 2]
                r2 = dx*dx + dy*dy + dz*dz + eps2
                inv_r = 1.0 / np.sqrt(r2)
                inv_r3 = inv_r / r2
                a[i, 0] += G * m[j] * dx * inv_r3
                a[i, 1] += G * m[j] * dy * inv_r3
                a[i, 2] += G * m[j] * dz * inv_r3

    # Kick velocities
    v[:] = v + 0.5 * (a_old + a) * dt
