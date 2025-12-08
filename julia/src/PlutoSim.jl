module PlutoSim

using StaticArrays
using LinearAlgebra
using Colors

export System, get_initial_conditions, rotate_pluto_inclination

# System struct to hold simulation state
mutable struct System
    num_particles::Int
    x::Matrix{Float64}  # positions [N×3]
    v::Matrix{Float64}  # velocities [N×3]
    m::Vector{Float64}  # masses
    G::Float64
end

# Gravitational acceleration calculation
function acceleration!(a, system::System)
    fill!(a, 0.0)
    x, m, G = system.x, system.m, system.G
    N = system.num_particles
    
    for i in 1:N
        for j in 1:N
            if i != j
                dx = x[j, 1] - x[i, 1]
                dy = x[j, 2] - x[i, 2]
                dz = x[j, 3] - x[i, 3]
                
                r_norm = sqrt(dx*dx + dy*dy + dz*dz)
                inv_r_cubed = 1.0 / (r_norm * r_norm * r_norm)
                
                a[i, 1] += G * m[j] * dx * inv_r_cubed
                a[i, 2] += G * m[j] * dy * inv_r_cubed
                a[i, 3] += G * m[j] * dz * inv_r_cubed
            end
        end
    end
end

# Velocity Verlet integrator
function velocity_verlet!(system::System, a::Matrix{Float64}, dt::Float64)
    a_old = copy(a)
    
    # Update positions: x = x + v*dt + 0.5*a*dt^2
    system.x .+= system.v .* dt .+ 0.5 .* a_old .* dt * dt
    
    # Calculate new accelerations
    acceleration!(a, system)
    
    # Update velocities: v = v + 0.5*(a_old + a)*dt
    system.v .+= 0.5 .* (a_old .+ a) .* dt
end

# Center of mass correction
function center_of_mass_correction!(system::System)
    M = sum(system.m)
    x_cm = sum(system.m[i] * system.x[i, :] for i in 1:system.num_particles) / M
    v_cm = sum(system.m[i] * system.v[i, :] for i in 1:system.num_particles) / M
    
    for i in 1:system.num_particles
        system.x[i, :] .-= x_cm
        system.v[i, :] .-= v_cm
    end
end

# Get initial conditions (solar system data)
function get_initial_conditions()
    # Conversion factor from km^3 s^-2 to AU^3 d^-2
    CONVERSION_FACTOR = (86400^2) / (149597870.7^3)
    
    # GM values in km^3 s^-2 (from NASA JPL)
    GM_KM_S = Dict(
        "Sun" => 132712440041.279419,
        "Mercury" => 22031.868551,
        "Venus" => 324858.592000,
        "Earth" => 398600.435507,
        "Mars" => 42828.375816,
        "Jupiter" => 126712764.100000,
        "Saturn" => 37940584.841800,
        "Uranus" => 5794556.400000,
        "Neptune" => 6836527.100580,
        "Pluto" => 975.500000,
        "Ceres" => 62.62890,
        "Vesta" => 17.288245
    )
    
    G = GM_KM_S["Sun"] * CONVERSION_FACTOR
    
    # Solar system masses (in M_sun)
    masses = [
        1.0,  # Sun
        GM_KM_S["Mercury"] / GM_KM_S["Sun"],
        GM_KM_S["Venus"] / GM_KM_S["Sun"],
        GM_KM_S["Earth"] / GM_KM_S["Sun"],
        GM_KM_S["Mars"] / GM_KM_S["Sun"],
        GM_KM_S["Jupiter"] / GM_KM_S["Sun"],
        GM_KM_S["Saturn"] / GM_KM_S["Sun"],
        GM_KM_S["Uranus"] / GM_KM_S["Sun"],
        GM_KM_S["Neptune"] / GM_KM_S["Sun"],
        GM_KM_S["Pluto"] / GM_KM_S["Sun"],
        GM_KM_S["Ceres"] / GM_KM_S["Sun"],
        GM_KM_S["Vesta"] / GM_KM_S["Sun"]
    ]
    
    # Positions (AU) - NASA JPL Horizons 2024-Jan-01
    positions = [
        [-7.967955691533730e-03, -2.906227441573178e-03, 2.103054301547123e-04],  # Sun
        [-2.825983269538632e-01, 1.974559795958082e-01, 4.177433558063677e-02],   # Mercury
        [-7.232103701666379e-01, -7.948302026312400e-02, 4.042871428174315e-02],  # Venus
        [-1.738192017257054e-01, 9.663245550235138e-01, 1.553901854897183e-04],   # Earth
        [-3.013262392582653e-01, -1.454029331393295e00, -2.300531433991428e-02],  # Mars
        [3.485202469657674e00, 3.552136904413157e00, -9.271035442798399e-02],     # Jupiter
        [8.988104223143450e00, -3.719064854634689e00, -2.931937777323593e-01],    # Saturn
        [1.226302417897505e01, 1.529738792480545e01, -1.020549026883563e-01],     # Uranus
        [2.983501460984741e01, -1.793812957956852e00, -6.506401132254588e-01],    # Neptune
        [1.720200478843485e01, -3.034155683573043e01, -1.729127607100611e00],     # Pluto
        [-1.103880510367569e00, -2.533340440444230e00, 1.220283937721780e-01],    # Ceres
        [-8.092549658731499e-02, 2.558381434460076e00, -6.695836142398572e-02]    # Vesta
    ]
    
    # Velocities (AU/day)
    velocities = [
        [4.875094764261564e-06, -7.057133213976680e-06, -4.573453713094512e-08],
        [-2.232165900189702e-02, -2.157207103176252e-02, 2.855193410495743e-04],
        [2.034068201002341e-03, -2.020828626592994e-02, -3.945639843855159e-04],
        [-1.723001232538228e-02, -2.967721342618870e-03, 6.382125383116755e-07],
        [1.424832259345280e-02, -1.579236181580905e-03, -3.823722796161561e-04],
        [-5.470970658852281e-03, 5.642487338479145e-03, 9.896190602066252e-05],
        [1.822013845554067e-03, 5.143470425888054e-03, -1.617235904887937e-04],
        [-3.097615358317413e-03, 2.276781932345769e-03, 4.860433222241686e-05],
        [1.676536611817232e-04, 3.152098732861913e-03, -6.877501095688201e-05],
        [2.802810313667557e-03, 8.492056438614633e-04, -9.060790113327894e-04],
        [8.978653480111301e-03, -4.873256528198994e-03, -1.807162046049230e-03],
        [-1.017876585480054e-02, -5.452367109338154e-04, 1.255870551153315e-03]
    ]
    
    # Labels and colors
    labels = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", 
              "Saturn", "Uranus", "Neptune", "Pluto", "Ceres", "Vesta"]
    
    colors = [:orange, :slategray, :wheat, :skyblue, :red, :darkgoldenrod,
              :gold, :paleturquoise, :blue, 
              RGB(0.635, 0.349, 0.902),  # Pluto purple
              RGB(0.902, 0.761, 0.161),  # Ceres golden
              RGB(0.349, 0.902, 0.510)]  # Vesta green
    
    x = reduce(hcat, positions)'  # Convert to N×3 matrix
    v = reduce(hcat, velocities)'
    
    system = System(length(masses), x, v, masses, G)
    center_of_mass_correction!(system)
    
    return system, labels, colors
end

# Rotate Pluto's orbit by inclination angle
function rotate_pluto_inclination!(system::System, orig_pos::Vector{Float64}, 
                                   orig_vel::Vector{Float64}, angle_deg::Float64)
    pluto_idx = 10  # Pluto is index 10
    angle = deg2rad(angle_deg)
    
    # Rotation matrix around x-axis
    rot_matrix = [1.0 0.0 0.0;
                  0.0 cos(angle) -sin(angle);
                  0.0 sin(angle) cos(angle)]
    
    system.x[pluto_idx, :] = rot_matrix * orig_pos
    system.v[pluto_idx, :] = rot_matrix * orig_vel
end

end # module
