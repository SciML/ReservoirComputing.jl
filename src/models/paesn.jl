@doc raw"""
    ParameterAwareESN(in_dims, res_dims, out_dims, activation=tanh;
        leak_coefficient=1.0, parameter_coupling=1.0, parameter_offset=0.0,
        init_reservoir=rand_sparse, init_input=scaled_rand, init_parameter=scaled_rand,
        init_bias=zeros32, init_state=randn32, use_bias=false,
        state_modifiers=(), readout_activation=identity)

Parameter-Aware Echo State Network (PA-ESN): a reservoir layer that incorporates
a control/bifurcation parameter into the dynamics, followed by optional state-modifier
layers and a linear readout.

The PA-ESN extends the standard ESN by adding an additional input channel for a
control parameter (e.g., bifurcation parameter), enabling the network to learn
dynamics across different parameter regimes. This allows the trained network to
predict system behavior at parameter values not seen during training.

`ParameterAwareESN` composes:
  1) a stateful [`ParameterAwareESNCell`](@ref) (reservoir with parameter input),
  2) zero or more `state_modifiers` applied to the reservoir state, and
  3) a [`LinearReadout`](@ref) mapping reservoir features to outputs.

## Equations

For input ``\mathbf{x}(t) \in \mathbb{R}^{in\_dims}``, control parameter
``\varepsilon(t) \in \mathbb{R}``, reservoir state
``\mathbf{h}(t) \in \mathbb{R}^{res\_dims}``, and output
``\mathbf{y}(t) \in \mathbb{R}^{out\_dims}``:

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \phi\!\left(\mathbf{W}_{in}\,\mathbf{x}(t) +
        \mathbf{W}_{res}\,\mathbf{h}(t-1) +
        k_b\,\mathbf{W}_b\,(\varepsilon(t) - \varepsilon_b) + \mathbf{b}\right) \\
    \mathbf{h}(t) &= (1-\alpha)\,\mathbf{h}(t-1) + \alpha\,\tilde{\mathbf{h}}(t) \\
    \mathbf{z}(t) &= \psi\!\left(\mathrm{Mods}\big(\mathbf{h}(t)\big)\right) \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{out}\,\mathbf{z}(t) + \mathbf{b}_{out}\right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension (state variables only, not including the control parameter).
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation (for [`ParameterAwareESNCell`](@ref)). Default: `tanh`.

## Keyword arguments

Reservoir (passed to [`ParameterAwareESNCell`](@ref)):

  - `leak_coefficient`: Leak rate ``\alpha \in (0,1]``. Default: `1.0`.
  - `parameter_coupling`: Parameter coupling strength ``k_b``. Default: `1.0`.
  - `parameter_offset`: Parameter offset ``\varepsilon_b``. Default: `0.0`.
  - `init_reservoir`: Initializer for ``\mathbf{W}_{res}``. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for ``\mathbf{W}_{in}``. Default: [`scaled_rand`](@ref).
  - `init_parameter`: Initializer for ``\mathbf{W}_b``. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initializer for reservoir bias (used if `use_bias=true`).
    Default: `zeros32`.
  - `init_state`: Initializer used when an external state is not provided.
    Default: `randn32`.
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.

Composition:

  - `state_modifiers`: A layer or collection of layers applied to the reservoir
    state before the readout. Accepts a single layer, an `AbstractVector`, or a
    `Tuple`. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default: `identity`.

## Inputs

  - `(x, param)`: A tuple where:
    - `x :: AbstractArray (in_dims, batch)` is the state input
    - `param :: AbstractArray (1, batch)` or scalar is the control parameter

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `reservoir` — parameters of the internal [`ParameterAwareESNCell`](@ref), including:
      - `input_matrix :: (res_dims × in_dims)` — ``\mathbf{W}_{in}``
      - `reservoir_matrix :: (res_dims × res_dims)` — ``\mathbf{W}_{res}``
      - `parameter_matrix :: (res_dims × 1)` — ``\mathbf{W}_b``
      - `bias :: (res_dims,)` — present only if `use_bias=true`
  - `states_modifiers` — a `Tuple` with parameters for each modifier layer (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × res_dims)` — ``\mathbf{W}_{out}``
      - `bias :: (out_dims,)` — ``\mathbf{b}_{out}`` (if the readout uses bias)

## States

  - `reservoir` — states for the internal [`ParameterAwareESNCell`](@ref).
  - `states_modifiers` — a `Tuple` with states for each modifier layer.
  - `readout` — states for [`LinearReadout`](@ref).

## References

  - Kong, L.-W., Weng, Y., Glaz, B., Haile, M., & Lai, Y.-C. (2021). Reservoir
    computing as digital twins for nonlinear dynamical systems. Chaos.
  - Patel, D., Canaday, D., Girvan, M., Pomerance, A., & Ott, E. (2021). Using
    machine learning to predict statistical properties of non-stationary
    dynamical processes. Physical Review E.

## Example

```julia
using ReservoirComputing, Random

# Create a parameter-aware ESN
rng = Random.default_rng()
paesn = ParameterAwareESN(3, 100, 3;
    parameter_coupling=0.5,
    parameter_offset=1.0)

ps, st = setup(rng, paesn)

# Input is a tuple of (state, parameter)
x = randn(Float32, 3, 1)  # state input
param = Float32[1.5;;]     # control parameter

y, st = paesn((x, param), ps, st)
```
"""
@concrete struct ParameterAwareESN <:
                 AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function ParameterAwareESN(in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        readout_activation = identity,
        state_modifiers = (),
        kwargs...)
    cell = ParameterAwareStatefulLayer(
        ParameterAwareESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return ParameterAwareESN(cell, mods, ro)
end

function Base.show(io::IO, paesn::ParameterAwareESN)
    print(io, "ParameterAwareESN(\n")

    print(io, "    reservoir = ")
    show(io, paesn.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(paesn.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(paesn.states_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, paesn.readout)
    print(io, "\n)")

    return
end

# Specialized collectstates for parameter-aware ESN
# Input data is expected as (state_data, param_data) where:
# - state_data is (in_dims, T)
# - param_data is (1, T) or (T,) - parameter values for each timestep
@doc raw"""
    collectstates(paesn::ParameterAwareESN, data::Tuple, ps, st)

Collect reservoir states from a parameter-aware ESN.

## Arguments

  - `paesn`: The ParameterAwareESN model.
  - `data`: A tuple `(state_data, param_data)` where:
    - `state_data :: AbstractMatrix (in_dims, T)` is the input sequence
    - `param_data :: AbstractMatrix (1, T)` or `AbstractVector (T,)` is the parameter sequence
  - `ps`: Model parameters.
  - `st`: Model states.

## Returns

  - `states`: Collected reservoir states of shape `(res_dims, T)`.
  - `st`: Updated model states.
"""
function collectstates(paesn::ParameterAwareESN,
        data::Tuple{<:AbstractMatrix, <:AbstractVecOrMat},
        ps, st::NamedTuple)
    state_data, param_data = data
    # Normalize param_data to be (1, T) matrix
    if ndims(param_data) == 1
        param_data = reshape(param_data, 1, :)
    end

    newst = st
    nsteps = size(state_data, 2)

    # Process first timestep
    x1 = state_data[:, 1]
    p1 = param_data[:, 1]
    current_state, partial_st = _partial_apply_paesn(paesn, (x1, p1), ps, newst)
    state_dims = size(current_state, 1)
    states = similar(state_data, state_dims, nsteps)
    states[:, 1] .= current_state
    newst = merge(partial_st, (readout = newst.readout,))

    # Process remaining timesteps
    for idx in 2:nsteps
        xi = state_data[:, idx]
        pi = param_data[:, idx]
        current_state, partial_st = _partial_apply_paesn(paesn, (xi, pi), ps, newst)
        states[:, idx] .= current_state
        newst = merge(partial_st, (readout = newst.readout,))
    end

    return states, newst
end

# Helper function for partial apply (reservoir + modifiers, no readout)
function _partial_apply_paesn(paesn::ParameterAwareESN, inp, ps, st)
    out, st_res = apply(paesn.reservoir, inp, ps.reservoir, st.reservoir)
    out,
    st_mods = _apply_seq(
        paesn.states_modifiers, out, ps.states_modifiers, st.states_modifiers)
    return out, (reservoir = st_res, states_modifiers = st_mods)
end

# Specialized train! for parameter-aware ESN
@doc raw"""
    train!(paesn::ParameterAwareESN, train_data, target_data, ps, st,
           train_method=StandardRidge(0.0); washout=0, return_states=false)

Train a parameter-aware ESN.

## Arguments

  - `paesn`: The ParameterAwareESN model.
  - `train_data`: A tuple `(state_data, param_data)` where:
    - `state_data :: AbstractMatrix (in_dims, T)` is the input sequence
    - `param_data :: AbstractMatrix (1, T)` or `AbstractVector (T,)` is the parameter sequence
  - `target_data`: Target output sequence of shape `(out_dims, T)`.
  - `ps`: Model parameters.
  - `st`: Model states.
  - `train_method`: Training method. Default: `StandardRidge(0.0)`.

## Keyword Arguments

  - `washout`: Number of initial timesteps to discard. Default: `0`.
  - `return_states`: If `true`, also return the collected states. Default: `false`.

## Returns

  - `(ps, st)`: Updated parameters and states.
  - `(ps, st), states`: If `return_states=true`.
"""
function train!(paesn::ParameterAwareESN,
        train_data::Tuple{<:AbstractMatrix, <:AbstractVecOrMat},
        target_data::AbstractMatrix, ps, st,
        train_method = StandardRidge(0.0);
        washout::Int = 0, return_states::Bool = false, kwargs...)
    states, st_after = collectstates(paesn, train_data, ps, st)
    states_wo,
    traindata_wo = washout > 0 ? _apply_washout(states, target_data, washout) :
                   (states, target_data)
    output_matrix = train(train_method, states_wo, traindata_wo; kwargs...)
    ps2, st_after = addreadout!(paesn, output_matrix, ps, st_after)
    return return_states ? ((ps2, st_after), states_wo) : (ps2, st_after)
end

# Specialized predict for parameter-aware ESN
@doc raw"""
    predict(paesn::ParameterAwareESN, data::Tuple, ps, st)
    predict(paesn::ParameterAwareESN, steps::Integer, ps, st;
            initialdata, initialparam, param_schedule)

Run prediction with a parameter-aware ESN.

## Teacher-forced mode

Given a tuple `(state_data, param_data)`, run the model in teacher-forced mode:

```julia
Y, st = predict(paesn, (state_data, param_data), ps, st)
```

## Auto-regressive mode

Roll the model forward for a fixed number of steps:

```julia
Y, st = predict(paesn, steps, ps, st;
    initialdata=x0,           # initial state vector
    initialparam=p0,          # initial parameter value (scalar or vector)
    param_schedule=schedule)  # function or vector giving parameter at each step
```

Where `param_schedule` can be:
- A function `f(t) -> param_value` giving the parameter at step `t`
- A vector of length `steps` with parameter values for each step
- A scalar if the parameter is constant

## Arguments

  - `paesn`: The ParameterAwareESN model.
  - `data`: Tuple `(state_data, param_data)` for teacher-forced mode.
  - `steps`: Number of steps for auto-regressive mode.
  - `ps`: Model parameters.
  - `st`: Model states.

## Keyword Arguments (auto-regressive mode only)

  - `initialdata`: Initial state vector of shape `(in_dims,)`.
  - `initialparam`: Initial parameter value.
  - `param_schedule`: Parameter schedule (function, vector, or scalar).

## Returns

  - `Y`: Output sequence of shape `(out_dims, T)` or `(out_dims, steps)`.
  - `st`: Updated model states.
"""
function predict(paesn::ParameterAwareESN,
        data::Tuple{<:AbstractMatrix, <:AbstractVecOrMat},
        ps, st)
    state_data, param_data = data
    # Normalize param_data
    if ndims(param_data) == 1
        param_data = reshape(param_data, 1, :)
    end

    T = size(state_data, 2)
    @assert T≥1 "data must have at least one time step (columns)."

    x1 = state_data[:, 1]
    p1 = param_data[:, 1]
    y1, st = apply(paesn, (x1, p1), ps, st)
    Y = similar(y1, size(y1, 1), T)
    Y[:, 1] .= y1

    for t in 2:T
        xt = state_data[:, t]
        pt = param_data[:, t]
        yt, st = apply(paesn, (xt, pt), ps, st)
        Y[:, t] .= yt
    end
    return Y, st
end

# Auto-regressive prediction
function predict(paesn::ParameterAwareESN, steps::Integer, ps, st;
        initialdata::AbstractVector,
        initialparam,
        param_schedule)
    T = eltype(initialdata)
    output = zeros(T, length(initialdata), steps)
    currentdata = initialdata

    for step in 1:steps
        # Get parameter for this step
        param = _get_param_at_step(param_schedule, step, initialparam, T)
        # Reshape param to (1,) vector if scalar
        param_vec = param isa Number ? T[param] : param
        currentdata, st = apply(paesn, (currentdata, param_vec), ps, st)
        output[:, step] = currentdata
    end
    return output, st
end

# Helper to get parameter at a given step
function _get_param_at_step(schedule::Function, step::Integer, ::Any, ::Type{T}) where {T}
    return T(schedule(step))
end

function _get_param_at_step(schedule::AbstractVector, step::Integer, ::Any, ::Type{T}) where {T}
    return T(schedule[step])
end

function _get_param_at_step(schedule::Number, ::Integer, ::Any, ::Type{T}) where {T}
    return T(schedule)
end

function _get_param_at_step(::Nothing, ::Integer, initialparam, ::Type{T}) where {T}
    return T(initialparam)
end
