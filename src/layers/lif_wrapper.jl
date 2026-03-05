@concrete struct LocalInformationFlow <: AbstractEchoStateNetworkCell
    cell
    lookback_horizon <: IntegerType
    init_buffer
end

function LocalInformationFlow((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        cell, lookback_horizon::IntegerType, args...; init_buffer = zeros32, kwargs...)
    built_cell = cell(in_dims => out_dims, args...; kwargs...)

    return LocalInformationFlow(built_cell, lookback_horizon, init_buffer)
end

function initialparameters(rng::AbstractRNG, lif::LocalInformationFlow)
    return (cell = initialparameters(rng, lif.cell),)
end

function initialstates(rng::AbstractRNG, lif::LocalInformationFlow)
    return (rng = sample_replicate(rng),
            input_buffer = nothing,
            cell = initialstates(rng, lif.cell))
end


function _init_input_buffer(rng::AbstractRNG, lif::LocalInformationFlow, inp::AbstractArray)
    n = max(lif.lookback_horizon - 1, 0)
    n == 0 && return ()
    buf = ntuple(_ -> similar(inp), n)
    in_dims = size(inp, 1)
    batch  = ndims(inp) == 1 ? 1 : size(inp, 2)
    for b in buf
        b .= lif.init_buffer(rng, in_dims, batch)
    end

    return buf
end

function init_hidden_states(rng::AbstractRNG, lif::LocalInformationFlow, inp::AbstractArray)
    return init_hidden_states(rng, lif.cell, inp)
end

function (lif::LocalInformationFlow)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    states  = init_hidden_states(rng, lif, inp)

    return lif((inp, states), ps, merge(st, (; rng)))
end

function (lif::LocalInformationFlow)((inp, states)::Tuple, ps, st::NamedTuple)
    buf = st.input_buffer
    if buf === nothing
        buf = _init_input_buffer(st.rng, lif, inp)
    end
    cell_ps = ps.cell
    cell_st = st.cell
    states_cur = states
    for inp_prev in buf
        (_, states_cur), cell_st = lif.cell((inp_prev, states_cur), cell_ps, cell_st)
    end
    (h_new, states_out), cell_st = lif.cell((inp, states_cur), cell_ps, cell_st)
    newbuf = isempty(buf) ? buf : (Base.tail(buf)..., inp)

    return (h_new, states_out), merge(st, (; input_buffer = newbuf, cell = cell_st))
end
