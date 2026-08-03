"""
    AbstractEchoStateNetwork{Fields} <: AbstractReservoirComputer{Fields}

Developer interface for an echo-state-network model container.

## Type parameters

- `Fields`: the Lux container field tuple. Most ESN models use
  `(:reservoir, :states_modifiers, :readout)`.

## Required fields

Subtypes follow the [`AbstractReservoirComputer`](@ref) field contract. Their
`reservoir` must be a Lux-compatible reservoir layer, normally a
[`StatefulLayer`](@ref) around an
[`AbstractEchoStateNetworkCell`](@ref). `states_modifiers` is a tuple of
feature transforms and `readout` is the final Lux-compatible mapping.

## Extension contract

This type adds ESN semantics to the generic reservoir-container contract; it
does not introduce a separate dispatch hook. Subtypes inherit the generic
`LuxCore.initialparameters`, `LuxCore.initialstates`, function-call, and
[`collectstates`](@ref) behavior when their fields obey the container
invariants. The reservoir must produce a consistent feature shape at every
time step, and the readout must consume that shape after all modifiers.

## Example

```julia
struct MyESN <: AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end
```
"""
abstract type AbstractEchoStateNetwork{Fields} <: AbstractReservoirComputer{Fields} end

_wrap_layer(x) = x isa Function ? WrappedFunction(x) : x
_wrap_layers(xs::Tuple) = map(_wrap_layer, xs)

@inline function _fillvec(x, n::Integer)
    v = Vector{typeof(x)}(undef, n)
    @inbounds @simd for i in 1:n
        v[i] = x
    end
    return v
end

@inline _asvec(::Tuple{}, n::Integer) = _fillvec(nothing, n)

@inline function _asvec(comp::Tuple, n::Integer)
    len = length(comp)
    if len == n
        return collect(comp)
    elseif len == 1
        return _fillvec(comp[1], n)
    else
        throw(DimensionMismatch("Expected length $n or 1, got $len"))
    end
end

@inline function _asvec(comp::AbstractVector, n::Integer)
    len = length(comp)
    if len == n
        return collect(comp)
    elseif len == 1
        return _fillvec(comp[1], n)
    else
        throw(DimensionMismatch("Expected length $n or 1, got $len"))
    end
end

@inline _asvec(::Nothing, n::Integer) = _fillvec(nothing, n)

@inline _asvec(comp, n::Integer) = _fillvec(comp, n)

@inline _asvec(x) = (ndims(x) == 2 ? vec(x) : x)

function _coerce_layer_mods(x)
    return x === nothing ? () :
        x isa Tuple ? x :
        x isa AbstractVector ? Tuple(x) :
        (x,)
end
