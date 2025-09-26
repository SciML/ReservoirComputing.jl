@concrete struct ESN <: AbstractLuxContainerLayer{(:cell, :states_modifiers, :readout)}
    cell
    states_modifiers
    readout
end

_wrap_layer(x) = x isa Function ? WrappedFunction(x) : x
_wrap_layers(xs::Tuple) = map(_wrap_layer, xs)

function ESN(in_dims::IntegerType, res_dims::IntegerType, out_dims::IntegerType, activation=tanh;
    readout_activation=identity,
    state_modifiers=(),
    kwargs...)
    cell = StatefulLayer(ESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return ESN(cell, mods, ro)
end

function initialparameters(rng::AbstractRNG, esn::ESN)
    ps_cell = initialparameters(rng, esn.cell)
    ps_mods = map(l -> initialparameters(rng, l), esn.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, esn.readout)
    return (cell=ps_cell, states_modifiers=ps_mods, readout=ps_ro)
end

function initialstates(rng::AbstractRNG, esn::ESN)
    st_cell = initialstates(rng, esn.cell)
    st_mods = map(l -> initialstates(rng, l), esn.states_modifiers) |> Tuple
    st_ro = initialstates(rng, esn.readout)
    return (cell=st_cell, states_modifiers=st_mods, readout=st_ro)
end

@inline function _apply_seq(layers::Tuple, x, ps::Tuple, st::Tuple)
    n = length(layers)
    new_st_parts = Vector{Any}(undef, n)
    @inbounds for i in 1:n
        x, sti = apply(layers[i], x, ps[i], st[i])
        new_st_parts[i] = sti
    end
    return x, tuple(new_st_parts...)
end

function (m::ESN)(x, ps, st)
    y, st_cell = apply(m.cell, x, ps.cell, st.cell)
    y, st_mods = _apply_seq(m.states_modifiers, y, ps.states_modifiers, st.states_modifiers)
    y, st_ro = apply(m.readout, y, ps.readout, st.readout)
    return y, (cell=st_cell, states_modifiers=st_mods, readout=st_ro)
end

function reset_carry(esn::ESN, st; mode=:zeros, value=nothing, rng=nothing)
    # Find current carry & infer shape/type
    c = get(st.cell, :carry, nothing)
    if c === nothing
        outd = esn.cell.cell.out_dims
        T = Float32
        sz = (outd, 1)
    else
        h = c[1]                 # carry is usually a 1-tuple (h,)
        T = eltype(h)
        sz = size(h)
    end

    new_h = begin
        if mode === :zeros
            zeros(T, sz)
        elseif mode === :randn
            rng = rng === nothing ? Random.default_rng() : rng
            randn(rng, T, sz...)
        elseif mode === :value
            @assert value !== nothing "Provide `value=` when mode=:value"
            fill(T(value), sz)
        else
            error("Unknown mode=$(mode). Use :zeros, :randn, or :value.")
        end
    end

    new_cell = merge(st.cell, (; carry=(new_h,)))
    return (cell=new_cell, states_modifiers=st.states_modifiers, readout=st.readout)
end

_set_readout_weight(ps_readout::NamedTuple, W) = merge(ps_readout, (; weight=W))

function train!(m::ESN, train_data::AbstractMatrix, target_data::AbstractMatrix,
    ps, st, train_method=StandardRidge(0.0);
    washout::Int=0, return_states::Bool=false)

    newst = st
    collected = Vector{Any}(undef, size(train_data, 2))
    @inbounds for (t, x) in enumerate(eachcol(train_data))
        y, st_cell = apply(m.cell, x, ps.cell, newst.cell)
        y, st_mods = _apply_seq(m.states_modifiers, y, ps.states_modifiers, newst.states_modifiers)
        collected[t] = copy(y)
        newst = (cell=st_cell, states_modifiers=st_mods, readout=newst.readout)
    end
    states = eltype(train_data).(reduce(hcat, collected))

    states_wo, targets_wo =
        washout > 0 ? _apply_washout(states, target_data, washout) : (states, target_data)

    W = train(train_method, states_wo, targets_wo)
    ps2 = (cell=ps.cell,
        states_modifiers=ps.states_modifiers,
        readout=_set_readout_weight(ps.readout, W))

    return return_states ? ((ps2, newst), states_wo) : (ps2, newst)
end

_basefuncstr(x) = sprint(show, x)

_getflag(x, sym::Symbol, default=false) = begin
    v = known(getproperty(x, Val(sym)))
    v === nothing ? default : v
end

function Base.show(io::IO, ::MIME"text/plain", rc::ReservoirChain)
    L = collect(pairs(rc.layers))
    if !isempty(L) && (L[1][2] isa StatefulLayer) && (L[end][2] isa LinearReadout)
        sl = L[1][2]
        ro = L[end][2]
        if sl.cell isa ESNCell
            esn = sl.cell
            mods = (length(L) > 2) ? map(x -> _basefuncstr(x[2]), L[2:end-1]) : String[]
            print(io, "ESN($(esn.in_dims) => $(esn.out_dims); ",
                "activation=", esn.activation,
                ", leak=", esn.leak_coefficient,
                ", readout=", ro.activation)
            ic = _getflag(ro, :include_collect, false)
            ic && print(io, ", include_collect=true")
            if !_getflag(esn, :use_bias, false)
                print(io, ", use_bias=false")
            end
            if !isempty(mods)
                print(io, ", modifiers=[", join(mods, ", "), "]")
            end
            print(io, ")")
            return
        end
    end
    strs = map(x -> _basefuncstr(x[2]), L)
    if length(strs) <= 2
        print(io, "ReservoirChain(", join(strs, ", "), ")")
    else
        print(io, "ReservoirChain(\n  ", join(strs, ",\n  "), "\n)")
    end
end

Base.show(io::IO, rc::ReservoirChain) = show(io, MIME"text/plain"(), rc)
