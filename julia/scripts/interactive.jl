using GLMakie
using Printf
include("../src/PlutoSim.jl")
using .PlutoSim

mutable struct SimulationState
    system::System
    labels::Vector{String}
    colors::Vector
    orig_pluto_pos::Vector{Float64}
    orig_pluto_vel::Vector{Float64}
    a::Matrix{Float64}
    trails::Vector{Vector{Vector{Float64}}}
    current_time::Float64
    dt::Float64
    steps_per_frame::Int
    trail_length::Int
    paused::Bool
    integrator_name::String
end

const INTEGRATORS = Dict(
    "Velocity-Verlet" => PlutoSim.velocity_verlet!,
    "Ruth-Forest"     => PlutoSim.ruth_forest!,
    "Yoshida-4"       => PlutoSim.yoshida4!,
)
const INTEGRATOR_NAMES = ["Velocity-Verlet", "Ruth-Forest", "Yoshida-4"]

function create_simulation()
    system, labels, colors = PlutoSim.get_initial_conditions()

    pluto_idx = 10
    orig_pluto_pos = copy(system.x[pluto_idx, :])
    orig_pluto_vel = copy(system.v[pluto_idx, :])

    a = zeros(system.num_particles, 3)
    PlutoSim.acceleration!(a, system)

    trails = [Vector{Vector{Float64}}() for _ in 1:system.num_particles]

    return SimulationState(
        system, labels, colors,
        orig_pluto_pos, orig_pluto_vel,
        a, trails,
        0.0,
        5.0,
        100,
        300,
        false,
        "Velocity-Verlet"
    )
end

function update_physics!(state::SimulationState)
    INTEGRATORS[state.integrator_name](state.system, state.a, state.dt)
    state.current_time += state.dt

    for i in 1:state.system.num_particles
        push!(state.trails[i], copy(state.system.x[i, :]))
        if length(state.trails[i]) > state.trail_length
            popfirst!(state.trails[i])
        end
    end
end

function run_interactive_simulation()
    println("Starting Interactive Solar System...")

    state = create_simulation()

    set_theme!(theme_black())
    fig = Figure(size=(1500, 1050), backgroundcolor=:black)

    ax = Axis3(fig[1, 1:3],
               xlabel="X (AU)", ylabel="Y (AU)", zlabel="Z (AU)",
               title="Interactive Solar System [Velocity-Verlet] - Time: 0.0 years",
               aspect=:data,
               backgroundcolor=(:black, 1.0),
               xgridcolor=(:white, 0.1),
               ygridcolor=(:white, 0.1),
               zgridcolor=(:white, 0.1),
               xlabelcolor=:white,
               ylabelcolor=:white,
               zlabelcolor=:white,
               titlecolor=:white,
               xticklabelcolor=:white,
               yticklabelcolor=:white,
               zticklabelcolor=:white)

    view_limit = 45.0
    limits!(ax, -view_limit, view_limit, -view_limit, view_limit, -view_limit, view_limit)

    body_positions = [Observable(Point3f(state.system.x[i, :]...)) for i in 1:state.system.num_particles]
    trail_data     = [Observable(Point3f[]) for i in 1:state.system.num_particles]
    time_text      = Observable("Time: 0.0 years (0 days)")

    marker_sizes = [30, 8, 12, 12, 10, 25, 23, 18, 18, 6, 5, 5]

    for i in 1:state.system.num_particles
        scatter!(ax, body_positions[i],
                color=state.colors[i],
                markersize=marker_sizes[i],
                label=state.labels[i],
                transparency=false)

        if i <= 9
            scatter!(ax, body_positions[i],
                    color=(state.colors[i], 0.3),
                    markersize=marker_sizes[i] * 1.8,
                    transparency=true)
        end
    end

    for i in 1:state.system.num_particles
        lines!(ax, trail_data[i],
              color=(state.colors[i], 0.6),
              linewidth=1.5)
    end

    Legend(fig[1, 4], ax,
           backgroundcolor=(:black, 0.8),
           labelcolor=:white,
           framecolor=(:white, 0.3),
           framewidth=1,
           labelsize=11)

    # ── Row 2: main sliders ──────────────────────────────────────────────
    sg = SliderGrid(fig[2, 1:3],
        (label="Speed (days/step)", range=1:1:20, startvalue=5,
         color_active=:orange, color_inactive=(:gray, 0.5)),
        width=Auto(),
        tellheight=true)
    speed_slider = sg.sliders[1]

    # ── Row 3: integrator buttons ────────────────────────────────────────
    integ_label = Label(fig[3, 1], "Integrator:",
                        color=:white, fontsize=13, halign=:right)

    integ_buttons = [
        Button(fig[3, 2], label=n,
               buttoncolor=n == "Velocity-Verlet" ? (:dodgerblue, 0.9) : (:gray30, 0.8),
               labelcolor=:white, fontsize=12)
        for n in INTEGRATOR_NAMES
    ]
    # Lay out the three buttons side by side in a sub-grid
    integ_grid = GridLayout(fig[3, 2:3])
    for (k, btn) in enumerate(integ_buttons)
        integ_grid[1, k] = btn
    end

    # ── Row 4: control buttons ────────────────────────────────────────────
    pause_button = Button(fig[4, 1], label="Pause",
                         buttoncolor=(:dodgerblue, 0.8),
                         labelcolor=:white, fontsize=14)
    reset_button = Button(fig[4, 2], label="Reset",
                         buttoncolor=(:orangered, 0.8),
                         labelcolor=:white, fontsize=14)
    experiments_button = Button(fig[4, 3], label="Experiments (show)",
                                buttoncolor=(:slategray, 0.8),
                                labelcolor=:white, fontsize=12)
    info_label = Label(fig[4, 4],
                      "Drag: Rotate | Scroll: Zoom",
                      color=:white, fontsize=12)

    # ── Row 5: collapsible experiments section ────────────────────────────
    experiments_visible = Observable(false)
    experiments_row = GridLayout(fig[5, 1:4])

    incl_sg = SliderGrid(experiments_row[1, 1:3],
        (label="Pluto Inclination (°)", range=0:1:90, startvalue=17,
         color_active=:skyblue, color_inactive=(:gray, 0.5)),
        width=Auto(), tellheight=true)
    inclination_slider = incl_sg.sliders[1]

    # Hide experiments row initially
    rowsize!(fig.layout, 5, 0)
    for el in experiments_row.content
        el.content.visible = false
    end

    # ── Callbacks ─────────────────────────────────────────────────────────

    on(experiments_button.clicks) do _
        experiments_visible[] = !experiments_visible[]
        if experiments_visible[]
            experiments_button.label = "Experiments (hide)"
            rowsize!(fig.layout, 5, Auto())
            for el in experiments_row.content
                el.content.visible = true
            end
        else
            experiments_button.label = "Experiments (show)"
            rowsize!(fig.layout, 5, 0)
            for el in experiments_row.content
                el.content.visible = false
            end
        end
    end

    on(inclination_slider.value) do val
        PlutoSim.rotate_pluto_inclination!(state.system, state.orig_pluto_pos,
                                          state.orig_pluto_vel, Float64(val))
        PlutoSim.acceleration!(state.a, state.system)
        empty!(state.trails[10])
        println("Pluto inclination set to $(val)°")
    end

    on(speed_slider.value) do val
        state.dt = Float64(val)
        println("Speed set to $(state.dt) days/step")
    end

    # Integrator selector
    for (k, btn) in enumerate(integ_buttons)
        local name = INTEGRATOR_NAMES[k]
        on(btn.clicks) do _
            state.integrator_name = name
            for (j, b) in enumerate(integ_buttons)
                b.buttoncolor = j == k ? (:dodgerblue, 0.9) : (:gray30, 0.8)
            end
            println("Integrator switched to $(name)")
        end
    end

    on(pause_button.clicks) do _
        state.paused = !state.paused
        if state.paused
            pause_button.label = "Resume"
            pause_button.buttoncolor = (:limegreen, 0.8)
        else
            pause_button.label = "Pause"
            pause_button.buttoncolor = (:dodgerblue, 0.8)
        end
        println(state.paused ? "Simulation paused" : "Simulation resumed")
    end

    on(reset_button.clicks) do _
        state.system, state.labels, state.colors = PlutoSim.get_initial_conditions()
        state.current_time = 0.0
        state.dt = 5.0
        state.paused = false
        state.trails = [Vector{Vector{Float64}}() for _ in 1:state.system.num_particles]
        PlutoSim.acceleration!(state.a, state.system)
        speed_slider.value[] = 5
        inclination_slider.value[] = 17
        pause_button.label = "Pause"
        pause_button.buttoncolor = (:dodgerblue, 0.8)
        println("Simulation reset")
    end

    # ── Animation loop ─────────────────────────────────────────────────────
    fps = 30

    function update_frame(_)
        if !state.paused
            for _ in 1:state.steps_per_frame
                update_physics!(state)
            end

            for i in 1:state.system.num_particles
                body_positions[i][] = Point3f(state.system.x[i, :]...)
                if length(state.trails[i]) > 1
                    trail_data[i][] = [Point3f(p...) for p in state.trails[i]]
                end
            end

            years = state.current_time / 365.25
            time_text[] = @sprintf("Time: %.1f years (%.0f days)", years, state.current_time)
            ax.title = "Interactive Solar System [$(state.integrator_name)] - " * time_text[]
        end
    end

    println("\n" * "="^60)
    println("JULIA INTERACTIVE SOLAR SYSTEM")
    println("="^60)
    println("Controls:")
    println("  - Drag mouse         -> Rotate 3D view")
    println("  - Scroll wheel       -> Zoom in/out")
    println("  - Integrator buttons -> Switch integrator at runtime")
    println("  - Speed slider       -> Change simulation speed")
    println("  - Pause button       -> Pause/Resume simulation")
    println("  - Reset button       -> Restart from beginning")
    println("  - Experiments button -> Show/hide Pluto inclination slider")
    println("="^60)
    println("Simulation starting...\n")

    display(fig)

    @async while isopen(fig.scene)
        update_frame(nothing)
        sleep(1/fps)
    end

    wait(fig.scene)
end

run_interactive_simulation()
