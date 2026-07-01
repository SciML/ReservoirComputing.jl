@doc raw"""
    ContinuousESNCell(in_dims => out_dims;
        tspan, args = (), activation = tanh, use_bias = false,
        init_bias = zeros32, init_reservoir = rand_sparse,
        init_input = scaled_rand, init_state = zeros32,
        equations = _continuous_esn_rhs!, kwargs...)

Continuous-time Echo State Network cell
([Lukosevicius2012](@cite)). Integrates the ODE

```math
\dot{\mathbf{x}}(t) = -\mathbf{x}(t) + \tanh\!\left(
    \mathbf{W}_{\text{in}}\,\mathbf{u}(t)
    + \mathbf{W}_r\,\mathbf{x}(t) + \mathbf{b}\right)
```

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir (hidden state) dimension.

## Keyword arguments

  - `tspan`: Integration interval `(t0, t1)`. Length-2, strictly
    increasing, finite.
  - `args`: Tuple of positional arguments forwarded to `solve`. The solver
    algorithm (`Tsit5()`, `Euler()`, …) is the first element by convention.
    Default: `()`.
  - `activation`: Reservoir activation. Default: `tanh`.
  - `use_bias`: Whether to include a bias term. Default: `false`.
  - `init_reservoir`: Initialiser for `W_r`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initialiser for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initialiser for the bias. Only used if `use_bias=true`.
    Default: `zeros32`.
  - `init_state`: Initialiser for the initial hidden state.
    Default: `zeros32`.
  - `equations`: ODE right-hand side `(dx, x, p, t) -> nothing`.
    Default: [`_continuous_esn_rhs!`](@ref).
  - `kwargs...`: Forwarded to `solve` as keyword arguments.

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_r`
  - `bias :: (out_dims,)` — present only if `use_bias = true`
"""
@concrete struct ContinuousESNCell <: AbstractSciMLProblemReservoir
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_state
    use_bias <: StaticBool
    equations
    tspan
    args
    kwargs
end

function ContinuousESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        tspan, args = (), activation = tanh, use_bias::BoolType = False(),
        init_bias = zeros32, init_reservoir = rand_sparse,
        init_input = scaled_rand, init_state = zeros32,
        equations = _continuous_esn_rhs!, kwargs...
    )
    return ContinuousESNCell(
        activation, in_dims, out_dims,
        init_bias, init_reservoir, init_input, init_state,
        static(use_bias),
        equations, tspan, args, kwargs
    )
end

function initialparameters(rng::AbstractRNG, cell::ContinuousESNCell)
    ps = (
        input_matrix = cell.init_input(rng, cell.out_dims, cell.in_dims),
        reservoir_matrix = cell.init_reservoir(rng, cell.out_dims, cell.out_dims),
    )
    if has_bias(cell)
        ps = merge(ps, (bias = cell.init_bias(rng, cell.out_dims),))
    end
    return ps
end

function initialstates(::AbstractRNG, ::ContinuousESNCell)
    return NamedTuple()
end

@doc raw"""
    _continuous_esn_rhs!(dx, x, p, t)

Default right-hand side for [`ContinuousESNCell`](@ref). Implements
[Lukosevicius2012](@cite) §3.2.6 eq (5):

```math
\dot{\mathbf{x}} = -\mathbf{x} + \tanh\!\left(
    \mathbf{W}_{\text{in}}\,\mathbf{u}(t)
    + \mathbf{W}_r\,\mathbf{x} + \mathbf{b}\right)
```

`p.input(t)` provides the interpolated input signal. The bias term is
included only when `p.bias` is present.
"""
function _continuous_esn_rhs!(dx, x, p, t)
    input_t = p.input(t)
    mul!(dx, p.reservoir_matrix, x)
    mul!(dx, p.input_matrix, input_t, true, true)
    if haskey(p, :bias)
        @. dx = -x + tanh(dx + p.bias)
    else
        @. dx = -x + tanh(dx)
    end
    return nothing
end

function Base.show(io::IO, cell::ContinuousESNCell)
    print(io, "ContinuousESNCell(", cell.in_dims, " => ", cell.out_dims)
    print(io, ", tspan = ")
    show(io, cell.tspan)
    has_bias(cell) || print(io, ", use_bias = false")
    return print(io, ")")
end
