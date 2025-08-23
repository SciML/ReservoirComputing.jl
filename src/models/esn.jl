function ESN(in_dims::IntegerType, res_dims::IntegerType, out_dims::IntegerType, activation=tanh;
    readout_activation=identity,
    state_modifiers=(),
    kwargs...)
    cell = ESNCell(in_dims => res_dims, activation; kwargs...)
    mods = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
           Tuple(state_modifiers) : (state_modifiers,)
    ro = Readout(res_dims => out_dims, readout_activation)
    return ReservoirChain((StatefulLayer(cell), mods..., ro)...)
end

_basefuncstr(x) = sprint(show, x)

_getflag(x, sym::Symbol, default=false) = begin
    v = known(getproperty(x, Val(sym)))
    v === nothing ? default : v
end

function Base.show(io::IO, ::MIME"text/plain", rc::ReservoirChain)
    L = collect(pairs(rc.layers))
    if !isempty(L) && (L[1][2] isa StatefulLayer) && (L[end][2] isa Readout)
        sl = L[1][2]
        ro = L[end][2]
        if sl.cell isa ESNCell
            esn = sl.cell
            # modifiers are anything between first and last
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
