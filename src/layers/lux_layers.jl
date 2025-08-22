abstract type AbstractReservoirCollectionLayer <: AbstractLuxLayer end
abstract type AbstractReservoirRecurrentCell <: AbstractLuxLayer end
abstract type AbstractReservoirTrainableLayer <: AbstractLuxLayer end


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

### Readout
# adapted from lux layers/basic Dense
@concrete struct Readout <: AbstractReservoirTrainableLayer
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    use_bias <: StaticBool
end

function Base.show(io::IO, d::Readout)
    print(io, "Readout($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    has_bias(d) || print(io, ", use_bias=false")
    return print(io, ")")
end

function Readout(mapping::Pair{<:IntegerType,<:IntegerType}, activation=identity; kwargs...)
    return Readout(first(mapping), last(mapping), activation; kwargs...)
end

function Readout(in_dims::IntegerType, out_dims::IntegerType, activation=identity;
    use_bias::BoolType=False())
    return Readout(activation, in_dims, out_dims, static(use_bias))
end

function initialparameters(rng::AbstractRNG, ro::Readout)
    weight = rand(rng, Float32, ro.out_dims, ro.in_dims)

    if has_bias(ro)
        return (; weight, bias=rand(rng, Float32, ro.out_dims))
    else
        return (; weight)
    end
end

parameterlength(ro::Readout) = ro.out_dims * ro.in_dims + has_bias(ro) * ro.out_dims
statelength(ro::Readout) = 0

outputsize(ro::Readout, _, ::AbstractRNG) = (ro.out_dims,)

function (ro::Readout)(inp::AbstractArray, ps, st::NamedTuple)
    out_tmp = ps.weight * inp
    if has_bias(ro)
        out_tmp += ps.bias
    end
    output = ro.activation.(out_tmp)
    return output, st
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

"""
    Collect()

A simple marker layer that passes data forward unchanged, but signals we want
to collect the passing states at this point.
"""
struct Collect <: AbstractReservoirCollectionLayer end

function (cl::Collect)(inp::AbstractArray, ps, st::NamedTuple)
    return inp, st
end

Base.show(io::IO, cl::Collect) = print(io, "Collection point of states")

"""
    collectstates(rc, data, ps, st)

Run `data` through the reservoir chain `rc` once and gather all states at every
`Collect()`. If more than one `Collect()` is present, the resulting vectors are
stacked with `vcat`.
"""
#=
function collectstates(rc::ReservoirChain, data::AbstractArray, ps, st::NamedTuple)
    collected = Any[]
    newst = (;)
    for inp in eachcol(data)
        state_tmp = Vector[]
        for (name, layer) in pairs(rc.layers)
            inp, st_tmp = layer(inp, ps[name], st[name])
            newst = merge(newst, (; name => st_tmp))
            if layer isa AbstractReservoirCollectionLayer
                state_tmp = vcat(state_tmp, inp)
            end
        end
        push!(collected, state_tmp)
    end
    return eltype(data).(reduce(hcat, collected))
end
=#
function collectstates(rc::ReservoirChain, data::AbstractArray, ps, st::NamedTuple)
    # Start from the incoming state and *carry it forward over time*
    newst = st
    collected = Any[]
    for inp_t in eachcol(data)
        x = inp_t
        state_vec = nothing

        for (name, layer) in pairs(rc.layers)
            x, st_i = layer(x, ps[name], newst[name])
            newst = merge(newst, (; name => st_i))
            if layer isa AbstractReservoirCollectionLayer
                state_vec = state_vec === nothing ? copy(x) : vcat(state_vec, x)
            end
        end
        push!(collected, state_vec === nothing ? copy(x) : state_vec)
    end
    return eltype(data).(reduce(hcat, collected))
end

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

# from Lux extended_ops
const KnownSymbolType{v} = Union{Val{v},StaticSymbol{v}}

function has_bias(l::AbstractLuxLayer)
    res = known(getproperty(l, Val(:use_bias)))
    return ifelse(res === nothing, false, res)
end

function getproperty(x, ::KnownSymbolType{v}) where {v}
    return v ∈ Base.propertynames(x) ? Base.getproperty(x, v) : nothing
end
