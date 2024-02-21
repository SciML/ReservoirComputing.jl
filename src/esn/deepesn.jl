struct DeepESN{I, S, N, T, O, M, B, ST, W, IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    nla_type::N
    input_matrix::T
    reservoir_driver::O
    reservoir_matrix::M
    bias_vector::B
    states_type::ST
    washout::W
    states::IS
end

function DeepESN(train_data,
        in_size::Int,
        res_size::Int;
        depth::Int=2,
        input_layer = fill(scaled_rand, depth),
        bias = fill(zeros64, depth),
        reservoir = fill(rand_sparse, depth),
        reservoir_driver = RNN(),
        nla_type = NLADefault(),
        states_type = StandardStates(),
        washout::Int = 0,
        rng = _default_rng(),
        T = Float64,
        matrix_type = typeof(train_data))
        
    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(Adapt.adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    end

    reservoir_matrix = [reservoir[i](rng, T, res_size, res_size) for i in 1:depth]
    input_matrix = [i == 1 ? input_layer[i](rng, T, res_size, in_size) : input_layer[i](rng, T, res_size, res_size) for i in 1:depth]
    bias_vector = [bias[i](rng, res_size) for i in 1:depth]
    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    DeepESN(res_size, train_data, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end
