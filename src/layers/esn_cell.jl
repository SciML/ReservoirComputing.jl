@concrete struct ESNCell <: AbstractReservoirRecurrentCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    #init_feedback::F
    init_state
    leak_coefficient
    use_bias <: StaticBool
end

function ESNCell((in_dims, out_dims)::Pair{<:Int,<:Int}, activation=tanh;
    use_bias::BoolType=False(), init_bias=zeros32, init_reservoir=rand_sparse,
    init_input=weighted_init, init_state=randn32, leak_coefficient=1.0)
    return ESNCell(activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_state, leak_coefficient, use_bias)
end

function initialparameters(rng::AbstractRNG, esn::ESNCell)
    ps = (input_matrix=esn.init_input(rng, esn.out_dims, esn.in_dims),
        reservoir_matrix=esn.init_reservoir(rng, esn.out_dims, esn.out_dims))
    if has_bias(esn)
        ps = merge(ps, (bias=esn.init_bias(rng, esn.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, esn::ESNCell)
    return (rng=sample_replicate(rng),)
end

function (esn::ESNCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, esn, inp)
    return esn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (esn::ESNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)
    if has_bias(esn)
        candidate_h = esn.activation.(ps.input_matrix * inp .+
                                      ps.reservoir_matrix * hidden_state .+ ps.bias)
    else
        candidate_h = esn.activation.(ps.input_matrix * inp .+
                                      ps.reservoir_matrix * hidden_state)
    end
    h_new = (T(1.0) - esn.leak_coefficient) .* hidden_state .+
            esn.leak_coefficient .* candidate_h
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, esn::ESNCell)
    print(io, "ESNCell($(esn.in_dims) => $(esn.out_dims)")
    if esn.leak_coefficient != eltype(esn.leak_coefficient)(1.0)
        print(io, ", leak_coefficient=$(esn.leak_coefficient)")
    end
    has_bias(esn) || print(io, ", use_bias=false")
    print(io, ")")
end
