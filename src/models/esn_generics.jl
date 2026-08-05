"""
    AbstractEchoStateNetwork{Fields} <: AbstractReservoirComputer{Fields}

Developer interface for an echo-state-network model container.

## Type parameters

- `Fields`: the Lux container field tuple. Most ESN models use
  `(:reservoir, :state_modifiers, :readout)`.

## Required fields

Subtypes follow the [`AbstractReservoirComputer`](@ref) field contract. Their
`reservoir` must be a Lux-compatible reservoir layer, normally a
[`StatefulLayer`](@ref) around an
[`AbstractEchoStateNetworkCell`](@ref). `state_modifiers` is a tuple of
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
struct MyESN <: AbstractEchoStateNetwork{(:reservoir, :state_modifiers, :readout)}
    reservoir
    state_modifiers
    readout
end
```
"""
abstract type AbstractEchoStateNetwork{Fields} <: AbstractReservoirComputer{Fields} end

__wrap_layer(x) = x isa Function ? WrappedFunction(x) : x
__wrap_layers(xs::Tuple) = map(__wrap_layer, xs)

@inline function __fillvec(x, n::Integer)
    v = Vector{typeof(x)}(undef, n)
    @inbounds @simd for i in 1:n
        v[i] = x
    end
    return v
end

@inline __asvec(::Tuple{}, n::Integer) = __fillvec(nothing, n)

@inline function __asvec(comp::Tuple, n::Integer)
    len = length(comp)
    if len == n
        return collect(comp)
    elseif len == 1
        return __fillvec(comp[1], n)
    else
        throw(DimensionMismatch("Expected length $n or 1, got $len"))
    end
end

@inline function __asvec(comp::AbstractVector, n::Integer)
    len = length(comp)
    if len == n
        return collect(comp)
    elseif len == 1
        return __fillvec(comp[1], n)
    else
        throw(DimensionMismatch("Expected length $n or 1, got $len"))
    end
end

@inline __asvec(::Nothing, n::Integer) = __fillvec(nothing, n)

@inline __asvec(comp, n::Integer) = __fillvec(comp, n)

@inline __asvec(x) = (ndims(x) == 2 ? vec(x) : x)

function __coerce_layer_mods(x)
    return x === nothing ? () :
        x isa Tuple ? x :
        x isa AbstractVector ? Tuple(x) :
        (x,)
end
