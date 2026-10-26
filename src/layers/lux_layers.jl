# from lux layers/basic
@concrete struct WrappedFunction <: AbstractLuxLayer
    func <: Function
end

(wf::WrappedFunction)(inp, ps, st::NamedTuple{}) = wf.func(inp), st

Base.show(io::IO, wf::WrappedFunction) = print(io, "WrappedFunction(", wf.func, ")")

# adapted from lux layers/recurrent StatefulRecurrentCell
@doc raw"""
    StatefulLayer(cell::AbstractReservoirRecurrentCell)

A lightweight wrapper that makes a recurrent cell carry its input state to the
next step.

## Arguments

- `cell`: `AbstractReservoirRecurrentCell` (e.g. [`ESNCell`](@ref)).

## States

- `cell`: internal states for the wrapped `cell` (e.g., RNG replicas, etc.).
- `carry`: the per-sequence hidden state; initialized to `nothing`.

"""
@concrete struct StatefulLayer <: AbstractLuxWrapperLayer{:cell}
    cell <: AbstractReservoirRecurrentCell
end

function initialstates(rng::AbstractRNG, sl::StatefulLayer)
    return (cell = initialstates(rng, sl.cell), carry = nothing)
end

function (sl::StatefulLayer)(inp, ps, st::NamedTuple)
    (out, carry), newst = applyrecurrentcell(sl.cell, inp, ps, st.cell, st.carry)
    return out, (; cell = newst, carry)
end

function applyrecurrentcell(sl::AbstractReservoirRecurrentCell, inp, ps, st, carry)
    return apply(sl, (inp, carry), ps, st)
end

function applyrecurrentcell(sl::AbstractReservoirRecurrentCell, inp, ps, st, ::Nothing)
    return apply(sl, inp, ps, st)
end

function Base.show(io::IO, sl::StatefulLayer)
    print(io, "StatefulLayer(")
    show(io, sl.cell)
    return print(io, ")")
end

@doc raw"""
    ReservoirChain(layers...; name=nothing)
    ReservoirChain(xs::AbstractVector; name=nothing)
    ReservoirChain(nt::NamedTuple; name=nothing)
    ReservoirChain(; name=nothing, kwargs...)

A lightweight, Lux-compatible container that composes a sequence of layers
and executes them in order. The implementation of `ReservoirChain` is
equivalent to Lux's own `Chain`.

## Construction

You can build a chain from:

  - **Positional layers:** `ReservoirChain(l1, l2, ...)`
  - **A vector of layers:** `ReservoirChain([l1, l2, ...])`
  - **A named tuple of layers:** `ReservoirChain((; layer_a=l1, layer_b=l2))`
  - **Keywords (sugar for a named tuple):** `ReservoirChain(; layer_a=l1, layer_b=l2)`

In all cases, function objects are automatically wrapped via `WrappedFunction`
so they can participate like regular layers. If a [`LinearReadout`](@ref) with
`include_collect=true` is present, the chain automatically inserts a [`Collect`](@ref)
layer immediately before that readout, unless a `Collect` already directly precedes
it. This holds for every constructor.

Use `name` to optionally tag the chain instance.

## Inputs

`(x, ps, st)` where:

  - `x`: input to the first layer.
  - `ps`: parameters as a named tuple with the same fields and order as the chain's layers.
  - `st`: states as a named tuple with the same fields and order as the chain's layers.

The call `(c::ReservoirChain)(x, ps, st)` forwards `x` through each layer:
`(x, ps_i, st_i) -> (x_next, st_i′)` and returns the final output and the
updated states for every layer.

## Returns

  - `(y, st′)` where `y` is the output of the last layer and `st′` is a named
    tuple collecting the updated states for each layer.

## Parameters

  - A `NamedTuple` whose fields correspond 1:1 with the layers. Each field
    holds the parameters for that layer.
  - Field names are generated as `:layer_1, :layer_2, ...` when constructed
    positionally, or preserved when you pass a `NamedTuple`/keyword constructor.
    A named entry `k` that expands to several layers (a [`LinearReadout`](@ref)
    with `include_collect=true` becomes `Collect()` followed by the readout,
    unless a `Collect` already precedes it) gives `:k_1, :k_2, ...` to the added
    layers and `:k` to the last one.

## States

  - A `NamedTuple` whose fields correspond 1:1 with the layers. Each field
    holds the state for that layer.

## Layer access & indexing

  - `c[i]`: get the *i*-th layer (1-based).
  - `c[indices]`: return a new `ReservoirChain` formed by selecting a subset of layers.
  - `getproperty(c, :layer_k)`: access layer `k` by its generated/explicit name.
  - `length(c)`, `firstindex(c)`, `lastindex(c)`: standard collection interfaces.

## Notes

  - **Function wrapping:** Any plain `Function` in the constructor is wrapped as
    `WrappedFunction(f)`. Non-layer, non-function objects will error.
  - **Auto-collect for readouts:** When a [`LinearReadout`](@ref) has
    `include_collect=true`, the constructor expands it to `(Collect(), readout)`
    so that downstream tooling can capture features consistently. No `Collect`
    is added when the layer right before the readout is already a `Collect`, so
    `ReservoirChain(reservoir, Collect(), LinearReadout(n => m))` has exactly one.

"""
@concrete struct ReservoirChain <: AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    name
end

function ReservoirChain(xs...; name = nothing)
    return ReservoirChain(named_tuple_layers(wrap_functions_in_chain_call(xs)...), name)
end
ReservoirChain(xs::AbstractVector; kwargs...) = ReservoirChain(xs...; kwargs...)
function ReservoirChain(nt::NamedTuple; name = nothing)
    names = Symbol[]
    layers = AbstractLuxLayer[]
    for (key, layer) in pairs(nt)
        start = length(layers)
        __wrap_into!(layers, (layer,))
        added = length(layers) - start
        for idx in 1:added
            push!(names, idx == added ? key : Symbol(key, :_, idx))
        end
    end
    allunique(names) || throw(
        ArgumentError("ReservoirChain layer names must be unique, got $(Tuple(names)).")
    )
    return ReservoirChain(NamedTuple{Tuple(names)}(Tuple(layers)), name)
end
ReservoirChain(; name = nothing, kwargs...) = ReservoirChain((; kwargs...); name)

function wrap_functions_in_chain_call(layers::Union{AbstractVector, Tuple})
    new_layers = []
    __wrap_into!(new_layers, layers)
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

wrap_functions_in_chain_call(x) = x

function __wrap_into!(acc, layers)
    for l in layers
        if l isa Tuple || l isa AbstractVector
            __wrap_into!(acc, l)
            continue
        end
        f = wrap_functions_in_chain_call(l)
        if f isa Tuple && !isempty(acc) && last(acc) isa Collect && first(f) isa Collect
            f = Base.tail(f)
        end
        for g in (f isa Tuple ? f : (f,))
            if g isa Function
                push!(acc, WrappedFunction(g))
            elseif g isa AbstractLuxLayer
                push!(acc, g)
            else
                throw(ArgumentError("Encountered a non-AbstractLuxLayer ($(typeof(g))) in ReservoirChain."))
            end
        end
    end
    return acc
end

function __readout_include_collect(ro::LinearReadout)
    res = known(getproperty(ro, Val(:include_collect)))
    return res === nothing ? false : res
end

function wrap_functions_in_chain_call(ro::LinearReadout)
    return __readout_include_collect(ro) ? (Collect(), ro) : ro
end

(c::ReservoirChain)(x, ps, st::NamedTuple) = applychain(c.layers, x, ps, st)

@generated function applychain(
        layers::NamedTuple{fields}, x, ps, st::NamedTuple{fields}
    ) where {fields}
    @assert isa(fields, NTuple{<:Any, Symbol})
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    st_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            (
                $(x_symbols[i + 1]),
                $(st_symbols[i]),
            ) = @inline apply(
                layers.$(fields[i]), $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i])
            )
        )
            for i in 1:N
    ]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(x_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.getindex(c::ReservoirChain, i::Int) = c.layers[i]
function Base.getindex(c::ReservoirChain, i::AbstractArray)
    return ReservoirChain(index_namedtuple(c.layers, i), nothing)
end

function Base.getproperty(c::ReservoirChain, name::Symbol)
    hasfield(typeof(c), name) && return getfield(c, name)
    layers = getfield(c, :layers)
    hasfield(typeof(layers), name) && return getfield(layers, name)
    throw(ArgumentError("$(typeof(c)) has no field or layer $name"))
end

Base.length(c::ReservoirChain) = length(c.layers)
Base.lastindex(c::ReservoirChain) = lastindex(c.layers)
Base.firstindex(c::ReservoirChain) = firstindex(c.layers)

function init_hidden_state(rng::AbstractRNG, rnn, inp)
    h0 = rnn.init_state(rng, rnn.out_dims, Base.size(inp, 2))
    T = promote_type(eltype(inp), eltype(h0))
    y = similar(inp, T, rnn.out_dims, Base.size(inp, 2))
    copyto!(y, h0)
    return ArrayInterface.aos_to_soa(y)
end

function named_tuple_layers(layers::Vararg{AbstractLuxLayer, N}) where {N}
    return NamedTuple{ntuple(i -> Symbol(:layer_, i), N)}(layers)
end

function index_namedtuple(nt::NamedTuple{fields}, idxs::AbstractArray) where {fields}
    return NamedTuple{fields[idxs]}(values(nt)[idxs])
end

# from Lux extended_ops
const KnownSymbolType{v} = Union{Val{v}, StaticSymbol{v}}

function has_bias(l::AbstractLuxLayer)
    res = known(getproperty(l, Val(:use_bias)))
    return ifelse(res === nothing, false, res)
end

@generated function getproperty(x::X, ::KnownSymbolType{v}) where {X, v}
    if hasfield(X, v)
        return :(getfield(x, v))
    else
        return :(nothing)
    end
end
