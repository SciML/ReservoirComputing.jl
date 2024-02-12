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
        res_size::AbstractArray;
        input_layer = scaled_rand,
        reservoir = rand_sparse,
        bias = zeros64,
        reservoir_driver = RNN(),
        nla_type = NLADefault(),
        states_type = StandardStates(),
        washout = 0,
        rng = _default_rng(),
        T = Float64,
        matrix_type = typeof(train_data))
        
    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(Adapt.adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    end

    reservoir_matrix = reservoir(rng, T, res_size, res_size)
    input_matrix = input_layer(rng, T, res_size, in_size)
    bias_vector = bias(rng, T, res_size)
    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    DeepESN(sum(res_size), train_data, variation, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end

function obtain_layers(in_size,
        input_layer,
        reservoir::Vector,
        bias;
        matrix_type = Matrix{Float64})
    esn_depth = length(reservoir)
    input_res_sizes = [get_ressize(reservoir[i]) for i in 1:esn_depth]
    in_sizes = zeros(Int, esn_depth)
    in_sizes[2:end] = input_res_sizes[1:(end - 1)]
    in_sizes[1] = in_size

    if input_layer isa Array
        input_matrix = [create_layer(input_layer[j], input_res_sizes[j], in_sizes[j],
            matrix_type = matrix_type) for j in 1:esn_depth]
    else
        _input_layer = fill(input_layer, esn_depth)
        input_matrix = [create_layer(_input_layer[k], input_res_sizes[k], in_sizes[k],
            matrix_type = matrix_type) for k in 1:esn_depth]
    end

    res_sizes = [get_ressize(input_matrix[j]) for j in 1:esn_depth]
    reservoir_matrix = [create_reservoir(reservoir[k], res_sizes[k],
        matrix_type = matrix_type) for k in 1:esn_depth]

    if bias isa Array
        bias_vector = [create_layer(bias[j], res_sizes[j], 1, matrix_type = matrix_type)
                       for j in 1:esn_depth]
    else
        _bias = fill(bias, esn_depth)
        bias_vector = [create_layer(_bias[k], res_sizes[k], 1, matrix_type = matrix_type)
                       for k in 1:esn_depth]
    end

    return input_matrix, reservoir_matrix, bias_vector, res_sizes
end
