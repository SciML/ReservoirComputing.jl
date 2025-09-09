# from lux layers/basic
@concrete struct WrappedFunction <: AbstractLuxLayer
    func <: Function
end

(wf::WrappedFunction)(inp, ps, st::NamedTuple{}) = wf.func(inp), st

Base.show(io::IO, wf::WrappedFunction) = print(io, "WrappedFunction(", wf.func, ")")

# adapted from lux layers/recurrent StatefulRecurrentCell
@concrete struct StatefulLayer <: AbstractLuxWrapperLayer{:cell}
    cell <: AbstractReservoirRecurrentCell
end

function initialstates(rng::AbstractRNG, sl::StatefulLayer)
    return (cell=initialstates(rng, sl.cell), carry=nothing)
end

function (sl::StatefulLayer)(inp, ps, st::NamedTuple)
    (out, carry), newst = applyrecurrentcell(sl.cell, inp, ps, st.cell, st.carry)
    return out, (; cell=newst, carry)
end

function applyrecurrentcell(sl::AbstractReservoirRecurrentCell, inp, ps, st, carry)
    return apply(sl, (inp, carry), ps, st)
end

function applyrecurrentcell(sl::AbstractReservoirRecurrentCell, inp, ps, st, ::Nothing)
    return apply(sl, inp, ps, st)
end

###build the ReservoirChain

#abstract type RCLayer <: AbstractLuxLayer end
#abstract type RCContainerLayer <: AbstractLuxContainerLayer end

"""
    ReservoirChain(layers...)

A simple container that holds a sequence of layers
"""
@concrete struct ReservoirChain <: AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    name
end

function ReservoirChain(xs...; name=nothing)
    return ReservoirChain(named_tuple_layers(wrap_functions_in_chain_call(xs)...), name)
end
ReservoirChain(xs::AbstractVector; kwargs...) = ReservoirChain(xs...; kwargs...)
ReservoirChain(nt::NamedTuple; name=nothing) = ReservoirChain(nt, name)
ReservoirChain(; name=nothing, kwargs...) = ReservoirChain((; kwargs...); name)

function wrap_functions_in_chain_call(layers::Union{AbstractVector,Tuple})
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

_readout_include_collect(ro::LinearReadout) = begin
    res = known(getproperty(ro, Val(:include_collect)))
    res === nothing ? false : res
end

function wrap_functions_in_chain_call(ro::LinearReadout)
    return _readout_include_collect(ro) ? (Collect(), ro) : ro
end

(c::ReservoirChain)(x, ps, st::NamedTuple) = applychain(c.layers, x, ps, st)

@generated function applychain(
    layers::NamedTuple{fields}, x, ps, st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(x_symbols[i+1]), $(st_symbols[i])) = @inline apply(
        layers.$(fields[i]), $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(x_symbols[N+1]), st))
    return Expr(:block, calls...)
end

Base.getindex(c::ReservoirChain, i::Int) = c.layers[i]
Base.getindex(c::ReservoirChain, i::AbstractArray) = ReservoirChain(index_namedtuple(c.layers, i))

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

function named_tuple_layers(layers::Vararg{AbstractLuxLayer,N}) where {N}
    return NamedTuple{ntuple(i -> Symbol(:layer_, i), N)}(layers)
end

function index_namedtuple(nt::NamedTuple{fields}, idxs::AbstractArray) where {fields}
    return NamedTuple{fields[idxs]}(values(nt)[idxs])
end

# from Lux extended_ops
const KnownSymbolType{v} = Union{Val{v},StaticSymbol{v}}

function has_bias(l::AbstractLuxLayer)
    res = known(getproperty(l, Val(:use_bias)))
    return ifelse(res === nothing, false, res)
end

@generated function getproperty(x::X, ::KnownSymbolType{v}) where {X,v}
    if hasfield(X, v)
        return :(getfield(x, v))
    else
        return :(nothing)
    end
end
