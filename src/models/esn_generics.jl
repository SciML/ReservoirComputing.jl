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
        error("Expected length $n or 1, got $len")
    end
end

@inline function _asvec(comp::AbstractVector, n::Integer)
    len = length(comp)
    if len == n
        return collect(comp)
    elseif len == 1
        return _fillvec(comp[1], n)
    else
        error("Expected length $n or 1, got $len")
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
