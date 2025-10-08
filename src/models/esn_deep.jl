"""
    DeepESN(in_dims::Int,
            res_dims::AbstractVector{<:Int},
            out_dims,
            activation=tanh;
            leak_coefficient=1.0,
            init_reservoir=rand_sparse,
            init_input=scaled_rand,
            init_bias=zeros32,
            init_state=randn32,
            use_bias=false,
            state_modifiers=(),
            readout_activation=identity)

Build a deep ESN: a stack of `StatefulLayer(ESNCell)` with optional per-layer
state modifiers, followed by a final linear readout.
"""
@concrete struct DeepESN <: AbstractEchoStateNetwork{(:cells, :states_modifiers, :readout)}
    cells
    states_modifiers
    readout
end

function DeepESN(in_dims::IntegerType,
    res_dims::AbstractVector{<:IntegerType},
    out_dims::IntegerType,
    activation=tanh;
    leak_coefficient=1.0,
    init_reservoir=rand_sparse,
    init_input=scaled_rand,
    init_bias=zeros32,
    init_state=randn32,
    use_bias=false,
    state_modifiers=(),
    readout_activation=identity)

    n_layers = length(res_dims)
    acts = _asvec(activation, n_layers)
    leaks = _asvec(leak_coefficient, n_layers)
    ires = _asvec(init_reservoir, n_layers)
    iinp = _asvec(init_input, n_layers)
    ibias = _asvec(init_bias, n_layers)
    ist = _asvec(init_state, n_layers)
    ub = _asvec(use_bias, n_layers)
    mods0 = _asvec(state_modifiers, n_layers)

    cells = Vector{Any}(undef, n_layers)
    states_modifiers = Vector{Any}(undef, n_layers)

    prev = in_dims
    for idx in firstindex(res_dims):lastindex(res_dims)
        cell = ESNCell(prev => res_dims[idx], acts[idx];
            use_bias=static(ub[idx]),
            init_bias=ibias[idx],
            init_reservoir=ires[idx],
            init_input=iinp[idx],
            init_state=ist[idx],
            leak_coefficient=leaks[idx])
        cells[idx] = StatefulLayer(cell)
        states_modifiers[idx] = mods0[idx] === nothing ? nothing : _wrap_layer(mods0[idx])
        prev = res_dims[idx]
    end
    mods_per_layer = map(_coerce_layer_mods, states_modifiers) |> Tuple
    ro = LinearReadout(prev => out_dims, readout_activation)
    return DeepESN(Tuple(cells), mods_per_layer, ro)
end

DeepESN(in_dims::Int, res_dim::Int, out_dims::Int; depth::Int=2, kwargs...) =
    DeepESN(in_dims, fill(res_dim, depth), out_dims; kwargs...)

function initialparameters(rng::AbstractRNG, desn::DeepESN)
    ps_cells = map(l -> initialparameters(rng, l), desn.cells) |> Tuple
    mods = desn.states_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
           desn.states_modifiers
    ps_mods = map(layer_mods ->
            (layer_mods === nothing ? () :
             map(l -> initialparameters(rng, l), layer_mods) |> Tuple),
        mods) |> Tuple

    ps_ro = initialparameters(rng, desn.readout)
    return (cells=ps_cells, states_modifiers=ps_mods, readout=ps_ro)
end

function initialstates(rng::AbstractRNG, desn::DeepESN)
    st_cells = map(l -> initialstates(rng, l), desn.cells) |> Tuple

    mods = desn.states_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
           desn.states_modifiers

    st_mods = map(layer_mods ->
            (layer_mods === nothing ? () :
             map(l -> initialstates(rng, l), layer_mods) |> Tuple),
        mods) |> Tuple

    st_ro = initialstates(rng, desn.readout)
    return (cells=st_cells, states_modifiers=st_mods, readout=st_ro)
end

function _partial_apply(desn::DeepESN, inp, ps, st)
    inp_t = inp
    n_layers = length(desn.cells)
    new_cell_st = Vector{Any}(undef, n_layers)
    new_mods_st = Vector{Any}(undef, n_layers)
    for idx in firstindex(desn.cells):lastindex(desn.cells)
        inp_t, st_cell_i = apply(desn.cells[idx], inp_t, ps.cells[idx], st.cells[idx])
        new_cell_st[idx] = st_cell_i
        inp_t, st_mods_i = _apply_seq(desn.states_modifiers[idx], inp_t,
            ps.states_modifiers[idx], st.states_modifiers[idx])
        new_mods_st[idx] = st_mods_i
    end

    return inp_t, (;
        cells=tuple(new_cell_st...),
        states_modifiers=tuple(new_mods_st...),
    )
end

function (desn::DeepESN)(inp, ps, st)
    out, new_st = _partial_apply(desn, inp, ps, st)
    inp_t, st_ro = apply(desn.readout, out, ps.readout, st.readout)
    return inp_t, merge(new_st, (readout=st_ro,))
end

function resetcarry!(rng::AbstractRNG, desn::DeepESN, st; init_carry=nothing)
    n_layers = length(desn.cells)

    @inline function _layer_outdim(idx)
        st_i = st.cells[idx]
        if st_i.carry === nothing
            return desn.cells[idx].cell.out_dims
        else
            return size(first(st_i.carry), 1)
        end
    end

    @inline function _init_for(idx)
        if init_carry === nothing
            return nothing
        elseif init_carry isa Function
            sz = _layer_outdim(idx)
            return (_asvec(init_carry(rng, sz)),)
        elseif init_carry isa Tuple || init_carry isa AbstractVector
            f = init_carry[idx]
            sz = _layer_outdim(idx)
            return f === nothing ? nothing : (_asvec(f(rng, sz)),)
        else
            throw(ArgumentError("init_carry must be nothing, a Function, or a Tuple/Vector of Functions"))
        end
    end

    new_cells = ntuple(idx -> begin
            st_i = st.cells[idx]
            new_carry = _init_for(idx)
            merge(st_i, (; carry=new_carry))
        end, n_layers)

    return (;
        cells=new_cells,
        states_modifiers=st.states_modifiers,
        readout=st.readout,
    )
end

function collectstates(desn::DeepESN, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    n_layers = length(desn.cells)
    for inp in eachcol(data)
        inp_t = inp
        cell_st_parts = Vector{Any}(undef, n_layers)
        mods_st_parts = Vector{Any}(undef, n_layers)
        for idx in firstindex(desn.cells):lastindex(desn.cells)
            inp_t, st_cell_i = apply(desn.cells[idx], inp_t, ps.cells[idx], newst.cells[idx])
            cell_st_parts[idx] = st_cell_i
            inp_t, st_mods_i = _apply_seq(
                desn.states_modifiers[idx], inp_t,
                ps.states_modifiers[idx], newst.states_modifiers[idx]
            )
            mods_st_parts[idx] = st_mods_i
        end
        push!(collected, copy(inp_t))
        newst = (;
            cells=tuple(cell_st_parts...),
            states_modifiers=tuple(mods_st_parts...),
            readout=newst.readout,
        )
    end
    @assert !isempty(collected)
    states = eltype(data).(reduce(hcat, collected))

    return states, newst
end

collectstates(m::DeepESN, data::AbstractVector, ps, st::NamedTuple) =
    collectstates(m, reshape(data, :, 1), ps, st)
