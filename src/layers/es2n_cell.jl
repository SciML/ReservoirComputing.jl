abstract type AbstractEchoStateNetworkCell <: AbstractReservoirRecurrentCell end

@doc raw"""
    ES2NCell(in_dims => out_dims, [activation];
        use_bias=False(), init_bias=zeros32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_state=randn32, init_orthogonal=orthogonal,
        proximity=1.0))

"""
@concrete struct ES2NCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_orthogonal
    init_state
    proximity
    use_bias <: StaticBool
end

function ES2NCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32, init_orthogonal = orthogonal,
        proximity::AbstractFloat = 1.0)
    return ES2NCell(activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_orthogonal, init_state, proximity, static(use_bias))
end

function initialparameters(rng::AbstractRNG, esn::ES2NCell)
    ps = (input_matrix = esn.init_input(rng, esn.out_dims, esn.in_dims),
        reservoir_matrix = esn.init_reservoir(rng, esn.out_dims, esn.out_dims),
        orthogonal_matrix = esn.init_orthogonal(rng, esn.out_dims, esn.out_dims))
    if has_bias(esn)
        ps = merge(ps, (bias = esn.init_bias(rng, esn.out_dims),))
    end
    return ps
end

function (esn::ES2NCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)
    if has_bias(esn)
        candidate_h = esn.activation.(ps.input_matrix * inp .+
                                      ps.reservoir_matrix * hidden_state .+ ps.bias)
    else
        candidate_h = esn.activation.(ps.input_matrix * inp .+
                                      ps.reservoir_matrix * hidden_state)
    end
    h_new = (T(1.0) - esn.proximity) .* ps.orthogonal_matrix * hidden_state .+
            esn.proximity .* candidate_h
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, esn::ES2NCell)
    print(io, "ES2NCell($(esn.in_dims) => $(esn.out_dims)")
    if esn.proximity != eltype(esn.proximity)(1.0)
        print(io, ", leak_coefficient=$(esn.proximity)")
    end
    has_bias(esn) || print(io, ", use_bias=false")
    print(io, ")")
end
