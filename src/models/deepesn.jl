# --- helpers ---
function _asvec(x, num_reservoirs::Int)
    if x === ()
        return ntuple(_ -> nothing, num_reservoirs)
    elseif x isa Tuple || x isa AbstractVector
        len = length(x)
        len == num_reservoirs && return Tuple(x)
        len == 1 && return ntuple(_ -> x[1], num_reservoirs)
        error("Expected length $num_reservoirs or 1 for per-layer argument, got $len")
    else
        return ntuple(_ -> x, num_reservoirs)
    end
end

function DeepESN(in_dims::Int,
    res_dims::AbstractVector{<:Int},
    out_dims,
    activation=tanh;
    leak_coefficient=1.0,
    init_reservoir=rand_sparse,
    init_input=weighted_init,
    init_bias=zeros32,
    init_state=randn32,
    use_bias=false,
    state_modifiers=(),
    readout_activation=identity)

    num_reservoirs = length(res_dims)

    acts = _asvec(activation, num_reservoirs)
    leaksv = _asvec(leak_coefficient, num_reservoirs)
    inres = _asvec(init_reservoir, num_reservoirs)
    ininp = _asvec(init_input, num_reservoirs)
    inbias = _asvec(init_bias, num_reservoirs)
    inst = _asvec(init_state, num_reservoirs)
    ubias = _asvec(use_bias, num_reservoirs)
    mods = _asvec(state_modifiers, num_reservoirs)

    layers = Any[]
    prev = in_dims
    for res in 1:num_reservoirs
        cell = ESNCell(prev => res_dims[res], acts[res];
            use_bias=static(ubias[res]),
            init_bias=inbias[res],
            init_reservoir=inres[res],
            init_input=ininp[res],
            init_state=inst[res],
            leak_coefficient=leaksv[res])

        push!(layers, StatefulLayer(cell))
        if mods[res] !== nothing
            push!(layers, mods[res])
        end
        prev = res_dims[res]
    end
    ro = Readout(prev => out_dims, readout_activation)
    return ReservoirChain((layers..., ro)...)
end

function DeepESN(in_dims::Int, res_dims::Int, out_dims::Int; depth::Int=2, kwargs...)
    return DeepESN(in_dims, fill(res_dims, depth), out_dims; kwargs...)
end
