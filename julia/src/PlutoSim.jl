module PlutoSim

using StaticArrays
using LinearAlgebra
using Colors

export System, get_initial_conditions, rotate_pluto_inclination, velocity_verlet!, ruth_forest!, yoshida4!, RECOMMENDED_DT

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

# Ruth-Forest 4th-order symplectic integrator
function ruth_forest!(system::System, a::Matrix{Float64}, dt::Float64)
    c = [7.0/24.0, 3.0/4.0, -1.0/24.0]
    d = [2.0/3.0, -2.0/3.0, 1.0]

    for s in 1:3
        system.x .+= c[s] .* system.v .* dt
        acceleration!(a, system)
        system.v .+= d[s] .* a .* dt
    end
end

# Yoshida 4th-order symplectic integrator
function yoshida4!(system::System, a::Matrix{Float64}, dt::Float64)
    cr = 2.0^(1.0/3.0)
    w0 = -cr / (2.0 - cr)
    w1 = 1.0 / (2.0 - cr)
    c = [w1, w0, w1]
    d = [w1/2.0, (w0 + w1)/2.0, (w0 + w1)/2.0, w1/2.0]

    for s in 1:3
        system.x .+= d[s] .* system.v .* dt
        acceleration!(a, system)
        system.v .+= c[s] .* a .* dt
    end
    # Final drift
    system.x .+= d[4] .* system.v .* dt
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

const RECOMMENDED_DT = Dict(
    "solar_system"       => 5.0,
    "solar_system_plus"  => 5.0,
    "solar_system_moons" => 0.1,
)

# Get initial conditions (solar system data)
function get_initial_conditions(initial_condition::String="solar_system_plus")
    CONVERSION_FACTOR = (86400^2) / (149597870.7^3)

    GM_KM_S = Dict(
        "Sun"      => 132712440041.279419,
        "Mercury"  => 22031.868551,
        "Venus"    => 324858.592000,
        "Earth"    => 398600.435507,
        "Mars"     => 42828.375816,
        "Jupiter"  => 126712764.100000,
        "Saturn"   => 37940584.841800,
        "Uranus"   => 5794556.400000,
        "Neptune"  => 6836527.100580,
        "Pluto"    => 975.500000,
        "Ceres"    => 62.62890,
        "Vesta"    => 17.288245,
        "Moon"     => 4902.800118,
        "Io"       => 5959.916,
        "Europa"   => 3202.739,
        "Ganymede" => 9887.834,
        "Callisto" => 7179.289,
        "Titan"    => 8978.137,
        "Triton"   => 1427.598,
        "Charon"   => 102.271,
    )

    G = GM_KM_S["Sun"] * CONVERSION_FACTOR

    ALL_POS = Dict(
        "Sun"      => [-7.967955691533730e-03, -2.906227441573178e-03,  2.103054301547123e-04],
        "Mercury"  => [-2.825983269538632e-01,  1.974559795958082e-01,  4.177433558063677e-02],
        "Venus"    => [-7.232103701666379e-01, -7.948302026312400e-02,  4.042871428174315e-02],
        "Earth"    => [-1.738192017257054e-01,  9.663245550235138e-01,  1.553901854897183e-04],
        "Mars"     => [-3.013262392582653e-01, -1.454029331393295e+00, -2.300531433991428e-02],
        "Jupiter"  => [ 3.485202469657674e+00,  3.552136904413157e+00, -9.271035442798399e-02],
        "Saturn"   => [ 8.988104223143450e+00, -3.719064854634689e+00, -2.931937777323593e-01],
        "Uranus"   => [ 1.226302417897505e+01,  1.529738792480545e+01, -1.020549026883563e-01],
        "Neptune"  => [ 2.983501460984741e+01, -1.793812957956852e+00, -6.506401132254588e-01],
        "Pluto"    => [ 1.720200478843485e+01, -3.034155683573043e+01, -1.729127607100611e+00],
        "Ceres"    => [-1.103880510367569e+00, -2.533340440444230e+00,  1.220283937721780e-01],
        "Vesta"    => [-8.092549658731499e-02,  2.558381434460076e+00, -6.695836142398572e-02],
        "Moon"     => [-1.762788124769829e-01,  9.674377513177153e-01,  3.236901585768862e-04],
        "Io"       => [-1.724522741438490e-01,  9.668498086618660e-01,  1.650795862021568e-04],
        "Europa"   => [-1.745785098571780e-01,  9.671303667029830e-01,  6.831340058359380e-05],
        "Ganymede" => [-1.750791193000750e-01,  9.674527498900250e-01, -2.154760302069210e-04],
        "Callisto" => [-1.711956677430700e-01,  9.643774044960820e-01, -1.290753714578770e-04],
        "Titan"    => [ 8.967025703553760e+00, -3.716783960302090e+00, -2.935428478015480e-01],
        "Triton"   => [ 2.983419878978100e+01, -1.797267195218900e+00, -6.507851303985160e-01],
        "Charon"   => [ 1.720203476697500e+01, -3.034152978258280e+01, -1.729133528455140e+00],
    )

    ALL_VEL = Dict(
        "Sun"      => [ 4.875094764261564e-06, -7.057133213976680e-06, -4.573453713094512e-08],
        "Mercury"  => [-2.232165900189702e-02, -2.157207103176252e-02,  2.855193410495743e-04],
        "Venus"    => [ 2.034068201002341e-03, -2.020828626592994e-02, -3.945639843855159e-04],
        "Earth"    => [-1.723001232538228e-02, -2.967721342618870e-03,  6.382125383116755e-07],
        "Mars"     => [ 1.424832259345280e-02, -1.579236181580905e-03, -3.823722796161561e-04],
        "Jupiter"  => [-5.470970658852281e-03,  5.642487338479145e-03,  9.896190602066252e-05],
        "Saturn"   => [ 1.822013845554067e-03,  5.143470425888054e-03, -1.617235904887937e-04],
        "Uranus"   => [-3.097615358317413e-03,  2.276781932345769e-03,  4.860433222241686e-05],
        "Neptune"  => [ 1.676536611817232e-04,  3.152098732861913e-03, -6.877501095688201e-05],
        "Pluto"    => [ 2.802810313667557e-03,  8.492056438614633e-04, -9.060790113327894e-04],
        "Ceres"    => [ 8.978653480111301e-03, -4.873256528198994e-03, -1.807162046049230e-03],
        "Vesta"    => [-1.017876585480054e-02, -5.452367109338154e-04,  1.255870551153315e-03],
        "Moon"     => [-1.746667306153906e-02, -3.473438277358121e-03, -3.359028758606074e-05],
        "Io"       => [-1.771011640498150e-02, -2.924774559185490e-03, -1.834718375093810e-05],
        "Europa"   => [-1.693671720285490e-02, -3.533618780072620e-03,  2.037063671680660e-05],
        "Ganymede" => [-1.641244918609050e-02, -3.208310050527570e-03,  8.449777989987570e-05],
        "Callisto" => [-1.524701777039010e-02, -2.380424571905720e-03,  1.284979573671200e-05],
        "Titan"    => [ 1.645916867278360e-03,  5.185540645697290e-03, -1.580268628283850e-04],
        "Triton"   => [ 2.183400296659120e-04,  3.135817255898950e-03, -6.881820099540450e-05],
        "Charon"   => [ 2.823416278266780e-03,  8.383578706697400e-04, -9.047649499895600e-04],
    )

    ALL_COLORS = Dict(
        "Sun"      => :orange,
        "Mercury"  => :slategray,
        "Venus"    => :wheat,
        "Earth"    => :skyblue,
        "Mars"     => :red,
        "Jupiter"  => :darkgoldenrod,
        "Saturn"   => :gold,
        "Uranus"   => :paleturquoise,
        "Neptune"  => :blue,
        "Pluto"    => RGB(0.635, 0.349, 0.902),
        "Ceres"    => RGB(0.902, 0.761, 0.161),
        "Vesta"    => RGB(0.349, 0.902, 0.510),
        "Moon"     => :lightgray,
        "Io"       => :khaki,
        "Europa"   => :lightcyan,
        "Ganymede" => :darkgray,
        "Callisto" => :dimgray,
        "Titan"    => :burlywood,
        "Triton"   => :lightsteelblue,
        "Charon"   => :gainsboro,
    )

    if initial_condition == "solar_system"
        names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"]
    elseif initial_condition == "solar_system_plus"
        names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
                 "Pluto","Ceres","Vesta"]
    elseif initial_condition == "solar_system_moons"
        names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
                 "Moon","Io","Europa","Ganymede","Callisto","Titan","Triton","Charon"]
    else
        error("Unknown initial condition: $initial_condition")
    end

    masses = [GM_KM_S[n] / GM_KM_S["Sun"] for n in names]
    positions = [ALL_POS[n] for n in names]
    velocities = [ALL_VEL[n] for n in names]
    labels = names
    colors = [ALL_COLORS[n] for n in names]

    x = reduce(hcat, positions)'
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
