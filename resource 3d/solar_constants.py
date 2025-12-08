# solar_constants.py

# Conversion factor from km^3 s^-2 to AU^3 d^-2
CONVERSION_FACTOR = (86400**2) / (149597870.7**3)

# GM values (km^3 s^-2)
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
}

# GM values (AU^3 d^-2)
GM_AU_DAY = {
    "Sun": 132712440041.279419 * CONVERSION_FACTOR,
    "Mercury": 22031.868551 * CONVERSION_FACTOR,
    "Venus": 324858.592000 * CONVERSION_FACTOR,
    "Earth": 398600.435507 * CONVERSION_FACTOR,
    "Mars": 42828.375816 * CONVERSION_FACTOR,
    "Jupiter": 126712764.100000 * CONVERSION_FACTOR,
    "Saturn": 37940584.841800 * CONVERSION_FACTOR,
    "Uranus": 5794556.400000 * CONVERSION_FACTOR,
    "Neptune": 6836527.100580 * CONVERSION_FACTOR,
}

# Solar system masses (M_sun^-1)
SOLAR_SYSTEM_MASSES = {
    "Sun": 1.0,
    "Mercury": GM_KM_S["Mercury"] / GM_KM_S["Sun"],
    "Venus": GM_KM_S["Venus"] / GM_KM_S["Sun"],
    "Earth": GM_KM_S["Earth"] / GM_KM_S["Sun"],
    "Mars": GM_KM_S["Mars"] / GM_KM_S["Sun"],
    "Jupiter": GM_KM_S["Jupiter"] / GM_KM_S["Sun"],
    "Saturn": GM_KM_S["Saturn"] / GM_KM_S["Sun"],
    "Uranus": GM_KM_S["Uranus"] / GM_KM_S["Sun"],
    "Neptune": GM_KM_S["Neptune"] / GM_KM_S["Sun"],
}

# Gravitational constant in AU^3 d^-2 M_sun^-1
G = GM_AU_DAY["Sun"]

# Solar system positions (AU)
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
}

# Solar system velocities (AU/day)
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
}

# Colors for visualization
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
