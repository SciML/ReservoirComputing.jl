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
    SVMReadout(first(mapping), last(mapping); kwargs...)
end

function SVMReadout(in_dims::IntegerType, out_dims::IntegerType;
        include_collect::BoolType = True())
    SVMReadout(in_dims, out_dims, static(include_collect))
end

initialparameters(::AbstractRNG, ::SVMReadout) = NamedTuple()
parameterlength(::SVMReadout) = 0
statelength(::SVMReadout) = 0
outputsize(ro::SVMReadout, _, ::AbstractRNG) = (ro.out_dims,)

# NOTE: forward for SVMReadout will be defined in the LIBSVM extension,
# because it calls LIBSVM.predict.

function _svmreadout_include_collect(ro::SVMReadout)
    ic = known(getproperty(ro, Val(:include_collect)))
    ic === nothing ? false : ic
end

function wrap_functions_in_chain_call(ro::SVMReadout)
    return _svmreadout_include_collect(ro) ? (Collect(), ro) : ro
end

_quote_keys(t) = Expr(:tuple, (QuoteNode(s) for s in t)...)

function _setmodels_rt(p::NamedTuple{K}, M) where {K}
    keys = K
    Kq = _quote_keys(keys)
    idx = findfirst(==(Symbol(:models)), keys)

    terms = Any[]
    for i in 1:length(keys)
        push!(terms, (idx === i) ? :(M) : :(getfield(p, $i)))
    end

    if idx === nothing
        newK = _quote_keys((keys..., :models))
        return :(NamedTuple{$newK}(($(terms...), M)))
    else
        return :(NamedTuple{$Kq}(($(terms...),)))
    end
end

@generated function _addsvm(layers::NamedTuple{K}, ps::NamedTuple{K}, M) where {K}
    if length(K) == 0
        return :(NamedTuple())
    end
    tailK = Base.tail(K)
    Kq = _quote_keys(K)
    tailKq = _quote_keys(tailK)

    head_val = :((getfield(layers, 1) isa SVMReadout)
                 ? _setmodels_rt(getfield(ps, 1), M)
                 : getfield(ps, 1))

    tail_call = :(_addsvm(NamedTuple{$tailKq}(Base.tail(layers)),
        NamedTuple{$tailKq}(Base.tail(ps)), M))

    return :(NamedTuple{$Kq}(($head_val, Base.values($tail_call)...)))
end

function addreadout!(rc::ReservoirChain, models, ps::NamedTuple, st::NamedTuple)
    @assert propertynames(rc.layers) == propertynames(ps)
    new_ps = _addsvm(rc.layers, ps, models)
    return new_ps, st
end
