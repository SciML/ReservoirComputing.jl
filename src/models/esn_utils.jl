abstract type AbstractEchoStateNetwork{Fields} <: AbstractReservoirComputer{Fields} end

_wrap_layer(x) = x isa Function ? WrappedFunction(x) : x
_wrap_layers(xs::Tuple) = map(_wrap_layer, xs)

@inline function _apply_seq(layers::Tuple, inp, ps::Tuple, st::Tuple)
    new_st_parts = Vector{Any}(undef, length(layers))
    for idx in eachindex(layers)
        inp, sti = apply(layers[idx], inp, ps[idx], st[idx])
        new_st_parts[idx] = sti
    end
    return inp, tuple(new_st_parts...)
end

function _asvec(comp, n_layers::Integer)
    if comp === ()
        return ntuple(_ -> nothing, n_layers)
    elseif comp isa Tuple || comp isa AbstractVector
        len = length(comp)
        if len == n_layers
            return Tuple(comp)
        elseif len == 1
            return ntuple(_ -> comp[1], n_layers)
        else
            error("Expected length $n_layers or 1, got $len")
        end
    else
        return ntuple(_ -> comp, n_layers)
    end
end

@inline _asvec(x) = (ndims(x) == 2 ? vec(x) : x)

_coerce_layer_mods(x) =
    x === nothing ? () :
    x isa Tuple ? x :
    x isa AbstractVector ? Tuple(x) :
    (x,)

_set_readout_weight(ps_readout::NamedTuple, wro) = merge(ps_readout, (; weight=wro))


function resetcarry(rng::AbstractRNG, esn, ps, st; init_carry=nothing)
    return ps, resetcarry(rng, esn, st; init_carry=init_carry)
end
