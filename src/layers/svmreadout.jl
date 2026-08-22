"""
    SVMReadout(in_dims => out_dims;
        include_collect=true, kwargs...)

Readout layer based on support vector machines. Requires LIBSVM.jl.

## Arguments

  - `in_dims`: Input/feature dimension (e.g., reservoir size).
  - `out_dims`: Output dimension (e.g., number of targets).

## Keyword arguments

  - `include_collect`: If `true` (default), training collects features immediately
    before this layer (as if a [`Collect()`](@ref) were inserted right before it).
  - `kwags`: specific keyword arguments for LIBSVM elements.
"""
@concrete struct SVMReadout <: AbstractReservoirTrainableLayer
    in_dims <: IntegerType
    out_dims <: IntegerType
    include_collect <: StaticBool
end

function Base.show(io::IO, ro::SVMReadout)
    print(io, "SVMReadout($(ro.in_dims) => $(ro.out_dims)")
    ic = known(getproperty(ro, Val(:include_collect)))
    ic === true && print(io, ", include_collect=true")
    return print(io, ")")
end

function SVMReadout(mapping::Pair{<:IntegerType, <:IntegerType}; kwargs...)
    return SVMReadout(first(mapping), last(mapping); kwargs...)
end

function SVMReadout(
        in_dims::IntegerType, out_dims::IntegerType;
        include_collect::BoolType = True()
    )
    return SVMReadout(in_dims, out_dims, static(include_collect))
end

initialparameters(::AbstractRNG, ::SVMReadout) = NamedTuple()
parameterlength(::SVMReadout) = 0
statelength(::SVMReadout) = 0
outputsize(ro::SVMReadout, _, ::AbstractRNG) = (ro.out_dims,)

# NOTE: forward for SVMReadout will be defined in the LIBSVM extension,
# because it calls LIBSVM.predict.

function __svmreadout_include_collect(ro::SVMReadout)
    ic = known(getproperty(ro, Val(:include_collect)))
    return ic === nothing ? false : ic
end

function wrap_functions_in_chain_call(ro::SVMReadout)
    return __svmreadout_include_collect(ro) ? (Collect(), ro) : ro
end

function addreadout!(rc::ReservoirChain, models, ps::NamedTuple, st::NamedTuple)
    @assert propertynames(rc.layers) == propertynames(ps)
    new_vals = map(Base.OneTo(length(propertynames(rc.layers)))) do i
        getfield(rc.layers, i) isa SVMReadout ? merge(getfield(ps, i), (models = models,)) :
            getfield(ps, i)
    end
    new_ps = NamedTuple{propertynames(rc.layers)}(Tuple(new_vals))
    return new_ps, st
end
