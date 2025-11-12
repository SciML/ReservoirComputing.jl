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
layer immediately before that readout.

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
    so that downstream tooling can capture features consistently.

"""
@concrete struct ReservoirChain <: AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    name::Any
end

function ReservoirChain(xs...; name = nothing)
    return ReservoirChain(named_tuple_layers(wrap_functions_in_chain_call(xs)...), name)
end
ReservoirChain(xs::AbstractVector; kwargs...) = ReservoirChain(xs...; kwargs...)
ReservoirChain(nt::NamedTuple; name = nothing) = ReservoirChain(nt, name)
ReservoirChain(; name = nothing, kwargs...) = ReservoirChain((; kwargs...); name)

function wrap_functions_in_chain_call(layers::Union{AbstractVector, Tuple})
    new_layers = []
    for l in layers
        f = wrap_functions_in_chain_call(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Function
            push!(new_layers, WrappedFunction(f))
        elseif f isa AbstractLuxLayer
            push!(new_layers, f)
        else
            throw("Encountered a non-AbstractLuxLayer in ReservoirChain.")
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

wrap_functions_in_chain_call(x) = x

function _readout_include_collect(ro::LinearReadout)
    res = known(getproperty(ro, Val(:include_collect)))
    res === nothing ? false : res
end

function wrap_functions_in_chain_call(ro::LinearReadout)
    return _readout_include_collect(ro) ? (Collect(), ro) : ro
end

(c::ReservoirChain)(x, ps, st::NamedTuple) = applychain(c.layers, x, ps, st)

@generated function applychain(
        layers::NamedTuple{fields}, x, ps, st::NamedTuple{fields}
) where {fields}
    @assert isa(fields, NTuple{<:Any, Symbol})
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(x_symbols[i + 1]),
                 $(st_symbols[i])) = @inline apply(
                 layers.$(fields[i]), $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(x_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.getindex(c::ReservoirChain, i::Int) = c.layers[i]
function Base.getindex(c::ReservoirChain, i::AbstractArray)
    ReservoirChain(index_namedtuple(c.layers, i))
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

### from Lux.Utils
function sample_replicate(rng::AbstractRNG)
    rand(rng)
    return replicate(rng)
end

function init_hidden_state(rng::AbstractRNG, rnn, inp)
    y = similar(inp, rnn.out_dims, Base.size(inp, 2))
    copyto!(y, rnn.init_state(rng, size(y)...))
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
