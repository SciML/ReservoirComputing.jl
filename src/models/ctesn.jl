@concrete struct CTESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function CTESN(
        in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        readout_activation = identity,
        state_modifiers = (),
        kwargs...
    )
    cell = StatefulLayer(CTESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return CTESN(cell, mods, ro)
end

function train!(
    rc::CTESN,
    input_data::AbstractMatrix,
    target_data::AbstractMatrix,
    ps,
    st,
    train_method = StandardRidge(0.0);
    washout::Int = 0,
    return_states::Bool = false,
    solver = Tsit5(),
    kwargs...
)
    cell = rc.reservoir.cell
    T = size(input_data, 2)

    # --- 1. Build continuous input u(t) via interpolation ---
    ts = collect(0:T-1)

    values = [Vector{eltype(input_data)}(input_data[:, i]) for i in 1:T]
    itp = LinearInterpolation(values, ts)

    u(t) = itp(clamp(t, ts[1], ts[end]))

    # Inject input into parameters (used inside ODE)
    ps_aug = (; ps..., input = u)

    # --- 2. Solve ODE once using your CTESNCell ---
    res_dim = size(ps.reservoir.reservoir_matrix, 1)
    x0 = zeros(eltype(input_data), res_dim)

    prob = ODEProblem(cell, x0, (ts[1], ts[end]), ps_aug)
    sol = solve(prob, solver)

    # --- 3. Sample reservoir states at discrete times ---
    states = hcat([sol(t) for t in ts]...)   # (res_dim, T)

    # --- 4. Apply washout (same as ESN pipeline) ---
    states_wo, targets_wo =
        washout > 0 ?
        _apply_washout(states, target_data, washout) :
        (states, target_data)

    # --- 5. Train readout ---
    output_matrix = train(train_method, states_wo, targets_wo; kwargs...)

    # --- 6. Attach readout ---
    ps2, st_after = addreadout!(rc, output_matrix, ps, st)

    # Store states for debugging/analysis
    st_after = merge(st_after, (; states = states_wo))

    return return_states ? ((ps2, st_after), states_wo) : (ps2, st_after)
end

# Forcing function
function predict(ctesn::CTESN, data::AbstractMatrix, ps, st;
                 solver=Tsit5())

    cell = ctesn.reservoir.cell
    T = size(data, 2)

    ts = collect(0:T-1)
    values = [Vector{eltype(data)}(data[:, i]) for i in 1:T]

    itp = LinearInterpolation(values, ts)

    function u(t)::Vector{eltype(data)}
        t_clamped = clamp(t, ts[1], ts[end])
        return itp(t_clamped)
    end

    ps = (; ps..., input = u)

    res_dim = size(ps.reservoir.reservoir_matrix, 1)
    x0 = zeros(eltype(data), res_dim)

    prob = ODEProblem(cell, x0, (0.0, T - 1), ps)
    sol = solve(prob, solver)
    
    y = t -> ps.readout.weight * sol(t)

    return y, sol.u[end]
end

# Autoregressive predict
function predict(
    ctesn::CTESN,
    steps::Int,
    ps,
    st;
    initialdata,
    solver = Tsit5()
)
    cell = ctesn.reservoir.cell

    res_dim = size(ps.reservoir.reservoir_matrix, 1)
    x0 = zeros(eltype(initialdata), res_dim)

    u0(t) = initialdata
    ps_init = (; ps..., input = u0)

    prob_init = ODEProblem(cell, x0, (0.0, 1.0), ps_init)
    sol_init = solve(prob_init, solver)

    x0 = sol_init.u[end]

    tspan = (0.0, steps - 1)
    prob = ODEProblem(cell, x0, tspan, ps)
    sol = solve(prob, solver)

    y = t -> ps.readout.weight * sol(t)

    return y, sol.u[end]
end