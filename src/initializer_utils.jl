module InitializerDefaults

    using Random: Xoshiro

    default_rng() = Xoshiro(1234)

end

module InitializerArrays

    using Random: AbstractRNG, rand!
    using WeightInitializers: zeros32

    function __array(rng::AbstractRNG, ::Type{T}, dims::Integer...) where {T}
        prototype = isempty(dims) ? zeros32(rng, 1) : zeros32(rng, dims...)
        return similar(prototype, T, dims)
    end

    function zeros(rng::AbstractRNG, ::Type{T}, dims::Integer...) where {T}
        return fill!(__array(rng, T, dims...), zero(T))
    end

    function ones(rng::AbstractRNG, ::Type{T}, dims::Integer...) where {T}
        return fill!(__array(rng, T, dims...), one(T))
    end

    function rand(rng::AbstractRNG, ::Type{T}, dims::Integer...) where {T}
        x = __array(rng, T, dims...)
        if T <: Complex
            rand!(rng, reinterpret(real(T), x))
        else
            rand!(rng, x)
        end
        return x
    end

end

module InitializerPartials

    using Random: AbstractRNG

    struct Partial{T, F, R, K} <: Function
        f::F
        rng::R
        kwargs::K
    end

    function Partial{T}(f, rng, kwargs) where {T}
        return Partial{T, typeof(f), typeof(rng), typeof(kwargs)}(f, rng, kwargs)
    end

    function (f::Partial{<:Union{Nothing, Missing}})(args...; kwargs...)
        f.rng === nothing && return f.f(args...; f.kwargs..., kwargs...)
        return f.f(f.rng, args...; f.kwargs..., kwargs...)
    end

    function (f::Partial{<:Union{Nothing, Missing}})(rng::AbstractRNG, args...; kwargs...)
        @assert f.rng === nothing
        return f.f(rng, args...; f.kwargs..., kwargs...)
    end

    function (f::Partial{T})(args...; kwargs...) where {T <: Number}
        f.rng === nothing && return f.f(T, args...; f.kwargs..., kwargs...)
        return f.f(f.rng, T, args...; f.kwargs..., kwargs...)
    end

    function (f::Partial{T})(rng::AbstractRNG, args...; kwargs...) where {T <: Number}
        @assert f.rng === nothing
        return f.f(rng, T, args...; f.kwargs..., kwargs...)
    end

end

const Utils = InitializerDefaults
const DeviceAgnostic = InitializerArrays
const PartialFunction = InitializerPartials
