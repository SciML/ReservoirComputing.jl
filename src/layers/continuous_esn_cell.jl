@doc raw"""
    ContinuousESNCell(activation, in_dims, out_dims,
                      init_bias, init_reservoir, init_input, init_state,
                      use_bias, equations, tspan, args, kwargs)

Continuous-time leaky-integrator ESN cell ([Lukosevicius2012](@cite) §3.2.6,
eq 5). `ContinuousESNCell <: AbstractSciMLProblemReservoir` so it dispatches
into the continuous `collectstates` / `predict` pipeline provided by the
`RCODEReservoirExt` package extension. Field layout mirrors [`ESNCell`](@ref)
plus the ODE right-hand side (`equations`) and the solve metadata
(`tspan`, `args`, `kwargs`) needed by the extension's solver.

The default `equations` value [`_continuous_esn_rhs!`](@ref) integrates
Lukoševičius 2012 §3.2.6 eq (5):

```math
\dot{\mathbf{x}}(t) = -\mathbf{x}(t) + \tanh\!\left(
    \mathbf{W}_{\text{in}}\,\mathbf{u}(t)
    + \mathbf{W}_r\,\mathbf{x}(t) + \mathbf{b}\right)
```

No leaking-rate term `α` appears in the ODE — `α` emerges only when the
ODE is forward-Euler discretised with step `Δt = α`, recovering the
discrete leaky update `x(n+1) = (1-α)·x(n) + α·tanh(...)`.

This type is normally constructed by the [`ContinuousESN`](@ref)
convenience constructor in the `RCODEReservoirExt` extension rather than
directly. Driving the inner constructor by hand requires placing every
field in the order above and pre-static-ing `use_bias`.

## Fields

  - `activation`: Reservoir nonlinearity. Documented as `tanh` by the
    `ContinuousESN` constructor; only enters the cell via a custom
    `equations` closure.
  - `in_dims`, `out_dims`: Input and reservoir dimensions.
  - `init_input`, `init_reservoir`, `init_bias`, `init_state`:
    Initialisers (same convention as [`ESNCell`](@ref)).
  - `use_bias::StaticBool`: Static-bool flag controlling the optional
    `bias` parameter.
  - `equations`: ODE right-hand side `(dx, x, p, t) -> nothing`. The
    extension injects `p.input(t)` (interpolated input signal) and merges
    `ps.reservoir` into `p`.
  - `tspan`: Integration interval `(t0, t1)` for `collectstates`.
  - `args`, `kwargs`: Forwarded to `solve` positionally / by keyword.

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_r`
  - `bias :: (out_dims,)` — present only if `use_bias = True()`
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
Lukoševičius 2012 §3.2.6 eq (5) in place:

```math
\dot{\mathbf{x}} = -\mathbf{x} + \tanh\!\left(
    \mathbf{W}_{\text{in}}\,\mathbf{u}(t)
    + \mathbf{W}_r\,\mathbf{x} + \mathbf{b}\right)
```

`p` is the merged parameter pack assembled by the `RCODEReservoirExt`
helper. It carries:
  * `p.input` — interpolated input signal injected by the extension.
  * `p.input_matrix` (`W_in`), `p.reservoir_matrix` (`W_r`) — reservoir
    matrices pulled from `ps.reservoir`.
  * `p.bias` (optional) — bias vector, present iff `use_bias=True()`.

The RHS uses `mul!` for both matrix products and a fused tanh broadcast
to stay allocation-free on the solver hot path. The bias branch is
selected at compile time via `haskey`, since the keys of a `NamedTuple`
are part of its type.
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
