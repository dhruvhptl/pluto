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
end

function create_simulation()
    system, labels, colors = PlutoSim.get_initial_conditions()
    
    # Store original Pluto vectors
    pluto_idx = 10
    orig_pluto_pos = copy(system.x[pluto_idx, :])
    orig_pluto_vel = copy(system.v[pluto_idx, :])
    
    # Initialize acceleration
    a = zeros(system.num_particles, 3)
    PlutoSim.acceleration!(a, system)
    
    # Initialize trails
    trails = [Vector{Vector{Float64}}() for _ in 1:system.num_particles]
    
    return SimulationState(
        system, labels, colors,
        orig_pluto_pos, orig_pluto_vel,
        a, trails,
        0.0,      # current_time
        5.0,      # dt (days)
        100,      # steps_per_frame
        300,      # trail_length
        false     # paused
    )
end

function update_physics!(state::SimulationState)
    PlutoSim.velocity_verlet!(state.system, state.a, state.dt)
    state.current_time += state.dt
    
    # Update trails
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
    
    # Create figure with dark space theme
    set_theme!(theme_black())
    fig = Figure(size=(1400, 1000), backgroundcolor=:black)
    ax = Axis3(fig[1, 1:3],
               xlabel="X (AU)", ylabel="Y (AU)", zlabel="Z (AU)",
               title="Interactive Solar System - Time: 0.0 years",
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
    
    # Set view limits
    view_limit = 45.0
    limits!(ax, -view_limit, view_limit, -view_limit, view_limit, -view_limit, view_limit)
    
    # Create observables for dynamic updates
    body_positions = [Observable(Point3f(state.system.x[i, :]...)) for i in 1:state.system.num_particles]
    trail_data = [Observable(Point3f[]) for i in 1:state.system.num_particles]
    time_text = Observable("Time: 0.0 years (0 days)")
    
    # Planet sizes (scaled for visibility)
    marker_sizes = [30, 8, 12, 12, 10, 25, 23, 18, 18, 6, 5, 5]
    
    # Plot bodies with glow effect
    for i in 1:state.system.num_particles
        # Main body
        scatter!(ax, body_positions[i], 
                color=state.colors[i], 
                markersize=marker_sizes[i],
                label=state.labels[i],
                transparency=false)
        
        # Glow effect for larger bodies
        if i <= 9  # Sun through Neptune
            scatter!(ax, body_positions[i], 
                    color=(state.colors[i], 0.3), 
                    markersize=marker_sizes[i] * 1.8,
                    transparency=true)
        end
    end
    
    # Plot trails with enhanced visibility
    for i in 1:state.system.num_particles
        lines!(ax, trail_data[i], 
              color=(state.colors[i], 0.6),
              linewidth=1.5)
    end
    
    # Add legend with styled background
    Legend(fig[1, 4], ax, 
           backgroundcolor=(:black, 0.8),
           labelcolor=:white,
           framecolor=(:white, 0.3),
           framewidth=1,
           labelsize=11)
    
    # Styled sliders
    sg = SliderGrid(fig[2, 1:3],
        (label="Pluto Inclination (°)", range=0:1:90, startvalue=17, 
         color_active=:skyblue, color_inactive=(:gray, 0.5)),
        (label="Speed (days/step)", range=1:1:20, startvalue=5, 
         color_active=:orange, color_inactive=(:gray, 0.5)),
        width=Auto(),
        tellheight=true)
    
    inclination_slider = sg.sliders[1]
    speed_slider = sg.sliders[2]
    
    # Styled buttons (no emojis)
    pause_button = Button(fig[3, 1], label="Pause", 
                         buttoncolor=(:dodgerblue, 0.8),
                         labelcolor=:white,
                         fontsize=14)
    
    reset_button = Button(fig[3, 2], label="Reset", 
                         buttoncolor=(:orangered, 0.8),
                         labelcolor=:white,
                         fontsize=14)
    
    # Info text box (no emojis)
    info_label = Label(fig[3, 3], 
                      "Drag: Rotate | Scroll: Zoom | Sliders: Control",
                      color=:white,
                      fontsize=12)
    
    # Slider callbacks
    on(inclination_slider.value) do val
        PlutoSim.rotate_pluto_inclination!(state.system, state.orig_pluto_pos, 
                                          state.orig_pluto_vel, Float64(val))
        PlutoSim.acceleration!(state.a, state.system)
        # Clear Pluto's trail
        empty!(state.trails[10])
        println("Pluto inclination set to $(val)°")
    end
    
    on(speed_slider.value) do val
        state.dt = Float64(val)
        println("Speed set to $(state.dt) days/step")
    end
    
    # Pause button callback
    on(pause_button.clicks) do n
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
    
    # Reset button callback
    on(reset_button.clicks) do n
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
    
    # Animation loop
    fps = 30
    
    # Update function
    function update_frame(t)
        if !state.paused
            # Multiple physics steps per frame
            for _ in 1:state.steps_per_frame
                update_physics!(state)
            end
            
            # Update observables
            for i in 1:state.system.num_particles
                body_positions[i][] = Point3f(state.system.x[i, :]...)
                
                if length(state.trails[i]) > 1
                    trail_data[i][] = [Point3f(p...) for p in state.trails[i]]
                end
            end
            
            # Update time display
            years = state.current_time / 365.25
            time_text[] = @sprintf("Time: %.1f years (%.0f days)", years, state.current_time)
            ax.title = "Interactive Solar System - " * time_text[]
        end
    end
    
    # Start animation
    println("\n" * "="^60)
    println("JULIA INTERACTIVE SOLAR SYSTEM")
    println("="^60)
    println("Controls:")
    println("  - Drag mouse       -> Rotate 3D view")
    println("  - Scroll wheel     -> Zoom in/out")
    println("  - Top slider       -> Adjust Pluto's inclination (0-90°)")
    println("  - Bottom slider    -> Change simulation speed")
    println("  - Pause button     -> Pause/Resume simulation")
    println("  - Reset button     -> Restart from beginning")
    println("="^60)
    println("Simulation starting...\n")
    
    display(fig)
    
    # Run animation loop
    @async while isopen(fig.scene)
        update_frame(nothing)
        sleep(1/fps)
    end
    
    wait(fig.scene)
end

# Run the simulation
run_interactive_simulation()
