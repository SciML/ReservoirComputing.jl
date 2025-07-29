abstract type AbstractReservoirDriver end

"""
    create_states(reservoir_driver::AbstractReservoirDriver, train_data, washout,
        reservoir_matrix, input_matrix, bias_vector)

Create and return the trained Echo State Network (ESN) states according to the
specified reservoir driver.

# Arguments

  - `reservoir_driver`: The reservoir driver that determines how the ESN states evolve
    over time.
  - `train_data`: The training data used to train the ESN.
  - `washout`: The number of initial time steps to discard during training to allow the
    reservoir dynamics to wash out the initial conditions.
  - `reservoir_matrix`: The reservoir matrix representing the dynamic, recurrent part of
    the ESN.
  - `input_matrix`: The input matrix that defines the connections between input features
    and reservoir nodes.
  - `bias_vector`: The bias vector to be added at each time step during the reservoir
    update.
"""
function create_states(reservoir_driver::AbstractReservoirDriver,
        train_data::AbstractArray, washout::Int, reservoir_matrix::AbstractMatrix,
        input_matrix::AbstractMatrix, bias_vector::AbstractArray)
    train_len = size(train_data, 2) - washout
    res_size = size(reservoir_matrix, 1)
    states = adapt(typeof(train_data), zeros(res_size, train_len))
    tmp_array = allocate_tmp(reservoir_driver, typeof(train_data), res_size)
    _state = adapt(typeof(train_data), zeros(res_size, 1))

    for i in 1:washout
        yv = @view train_data[:, i]
        _state = next_state!(_state, reservoir_driver, _state, yv, reservoir_matrix,
            input_matrix, bias_vector, tmp_array)
    end

    for j in 1:train_len
        yv = @view train_data[:, washout + j]
        _state = next_state!(_state, reservoir_driver, _state, yv,
            reservoir_matrix, input_matrix, bias_vector, tmp_array)
        states[:, j] = _state
    end

    return states
end

function create_states(reservoir_driver::AbstractReservoirDriver,
        train_data::AbstractArray, washout::Int, reservoir_matrix::Vector,
        input_matrix::AbstractArray, bias_vector::AbstractArray)
    train_len = size(train_data, 2) - washout
    res_size = sum([size(reservoir_matrix[i], 1) for i in 1:length(reservoir_matrix)])
    states = adapt(typeof(train_data), zeros(res_size, train_len))
    tmp_array = allocate_tmp(reservoir_driver, typeof(train_data), res_size)
    _state = adapt(typeof(train_data), zeros(res_size))

    for i in 1:washout
        for j in 1:length(reservoir_matrix)
            _inter_state = next_state!(_inter_state, reservoir_driver, _inter_state,
                train_data[:, i],
                reservoir_matrix, input_matrix, bias_vector,
                tmp_array)
        end
        _state = next_state!(_state, reservoir_driver, _state, train_data[:, i],
            reservoir_matrix, input_matrix, bias_vector, tmp_array)
    end

    for j in 1:train_len
        _state = next_state!(_state, reservoir_driver, _state, train_data[:, washout + j],
            reservoir_matrix, input_matrix, bias_vector, tmp_array)
        states[:, j] = _state
    end

    return states
end

#standard RNN driver
struct RNN{F, T} <: AbstractReservoirDriver
    activation_function::F
    leaky_coefficient::T
end

"""
    RNN(activation_function, leaky_coefficient)
    RNN(;activation_function=tanh, leaky_coefficient=1.0)

Returns a Recurrent Neural Network (RNN) initializer for
echo state networks (`ESN`).

# Arguments

  - `activation_function`: The activation function used in the RNN.
  - `leaky_coefficient`: The leaky coefficient used in the RNN.

# Keyword Arguments

  - `activation_function`: The activation function used in the RNN.
    Defaults to `tanh_fast`.
  - `leaky_coefficient`: The leaky coefficient used in the RNN.
    Defaults to 1.0.
"""
function RNN(; activation_function = fast_act(tanh), leaky_coefficient = 1.0)
    return RNN(activation_function, leaky_coefficient)
end

function reservoir_driver_params(rnn::RNN, args...)
    return rnn
end

function next_state!(out, rnn::RNN, x, y, W, W_in, b, tmp_array)
    mul!(tmp_array[1], W, x)
    mul!(tmp_array[2], W_in, y)
    @. tmp_array[1] = rnn.activation_function(tmp_array[1] + tmp_array[2] + b) *
                      rnn.leaky_coefficient
    return @. out = (1 - rnn.leaky_coefficient) * x + tmp_array[1]
end

function next_state!(out, rnn::RNN, x, y, W::Vector, W_in, b, tmp_array)
    esn_depth = length(W)
    res_sizes = vcat(0, [size(W[i], 1) for i in 1:esn_depth])
    inner_states = [x[(1 + sum(res_sizes[1:i])):sum(res_sizes[1:(i + 1)])]
                    for i in 1:esn_depth]
    inner_inputs = vcat([y], inner_states[1:(end - 1)])

    for i in 1:esn_depth
        inner_states[i] = (1 - rnn.leaky_coefficient) .* inner_states[i] +
                          rnn.leaky_coefficient *
                          rnn.activation_function.((W[i] * inner_states[i]) .+
                                                   (W_in[i] * inner_inputs[i]) .+
                                                   reduce(vcat, b[i]))
    end
    return reduce(vcat, inner_states)
end

function allocate_tmp(::RNN, tmp_type, res_size)
    return [adapt(tmp_type, zeros(res_size, 1)) for i in 1:2]
end

#multiple RNN driver
struct MRNN{F, T, R} <: AbstractReservoirDriver
    activation_function::F
    leaky_coefficient::T
    scaling_factor::R
end

"""
    MRNN(activation_function, leaky_coefficient, scaling_factor)
    MRNN(;activation_function=[tanh, sigmoid], leaky_coefficient=1.0,
        scaling_factor=fill(leaky_coefficient, length(activation_function)))

Returns a Multiple RNN (MRNN) initializer for the Echo State Network (ESN),
introduced in [Lun2015](@cite).

# Arguments

  - `activation_function`: A vector of activation functions used
    in the MRNN.
  - `leaky_coefficient`: The leaky coefficient used in the MRNN.
  - `scaling_factor`: A vector of scaling factors for combining activation
    functions.

# Keyword Arguments

  - `activation_function`: A vector of activation functions used in the MRNN.
    Defaults to `[tanh, sigmoid]`.
  - `leaky_coefficient`: The leaky coefficient used in the MRNN.
    Defaults to 1.0.
  - `scaling_factor`: A vector of scaling factors for combining activation functions.
    Defaults to an array of the same size as `activation_function` with all
    elements set to `leaky_coefficient`.

This function creates an MRNN object with the specified activation functions,
leaky coefficient, and scaling factors, which can be used as a reservoir driver
in the ESN.
"""
function MRNN(; activation_function = [tanh, sigmoid],
        leaky_coefficient = 1.0,
        scaling_factor = fill(leaky_coefficient, length(activation_function)))
    @assert length(activation_function) == length(scaling_factor)
    return MRNN(activation_function, leaky_coefficient, scaling_factor)
end

function reservoir_driver_params(mrnn::MRNN, args...)
    return mrnn
end

function next_state!(out, mrnn::MRNN, x, y, W, W_in, b, tmp_array)
    @. out = (1 - mrnn.leaky_coefficient) * x
    for i in 1:length(mrnn.scaling_factor)
        mul!(tmp_array[1], W, x)
        mul!(tmp_array[2], W_in, y)
        @. out += mrnn.activation_function[i](tmp_array[1] + tmp_array[2] + b) *
                  mrnn.scaling_factor[i]
    end

    return out
end

function allocate_tmp(::MRNN, tmp_type, res_size)
    return [adapt(tmp_type, zeros(res_size, 1)) for i in 1:2]
end

abstract type AbstractGRUVariant end
#GRU-based driver
struct GRU{F, L, R, V, B} #not an abstractreservoirdriver
    activation_function::F
    inner_layer::L
    reservoir::R
    bias::B
    variant::V
end

#https://arxiv.org/abs/1701.05923# variations of gru
"""
    FullyGated()

Returns a Fully Gated Recurrent Unit (FullyGated) initializer
for the Echo State Network (ESN).

Returns the standard gated recurrent unit [Cho2014](@cite) as a driver for the
echo state network (`ESN`).
"""
struct FullyGated <: AbstractGRUVariant end

"""
    Minimal()

Returns a minimal GRU ESN initializer.
"""
struct Minimal <: AbstractGRUVariant end

#layer_init and activation_function must be vectors
"""
    GRU(;activation_function=[NNlib.sigmoid, NNlib.sigmoid, tanh],
        inner_layer = fill(DenseLayer(), 2),
        reservoir = fill(RandSparseReservoir(), 2),
        bias = fill(DenseLayer(), 2),
        variant = FullyGated())

Returns a Gated Recurrent Unit (GRU) reservoir driver for Echo State Network (`ESN`).
This driver is based on the GRU architecture [Cho2014](@cite).

# Arguments

  - `activation_function`: An array of activation functions for the GRU layers.
    By default, it uses sigmoid activation functions for the update gate, reset gate,
    and tanh for the hidden state.
  - `inner_layer`: An array of inner layers used in the GRU architecture.
    By default, it uses two dense layers.
  - `reservoir`: An array of reservoir layers.
    By default, it uses two random sparse reservoirs.
  - `bias`: An array of bias layers for the GRU.
    By default, it uses two dense layers.
  - `variant`: The GRU variant to use.
    By default, it uses the "FullyGated" variant.
"""
function GRU(; activation_function = [sigmoid, sigmoid, tanh],
        inner_layer = fill(scaled_rand, 2),
        reservoir = fill(rand_sparse, 2),
        bias = fill(scaled_rand, 2),
        variant = FullyGated())
    return GRU(activation_function, inner_layer, reservoir, bias, variant)
end

#the actual params are only available inside ESN(), so a different driver is needed
struct GRUParams{F, V, S, I, N, SF, IF, NF} <: AbstractReservoirDriver
    activation_function::F
    variant::V
    Wz_in::S
    Wz::I
    bz::N
    Wr_in::SF
    Wr::IF
    br::NF
end

#vreation of the actual driver
function reservoir_driver_params(gru::GRU, res_size, in_size)
    gru_params = create_gru_layers(gru, gru.variant, res_size, in_size)
    return gru_params
end

#dispatch on the different gru variations
function create_gru_layers(gru, variant::FullyGated, res_size, in_size)
    Wz_in = gru.inner_layer[1](res_size, in_size)
    Wz = gru.reservoir[1](res_size, res_size)
    bz = gru.bias[1](res_size, 1)

    Wr_in = gru.inner_layer[2](res_size, in_size)
    Wr = gru.reservoir[2](res_size, res_size)
    br = gru.bias[2](res_size, 1)

    return GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br)
end

#check this one, not sure
function create_gru_layers(gru, variant::Minimal, res_size, in_size)
    Wz_in = gru.inner_layer(res_size, in_size)
    Wz = gru.reservoir(res_size, res_size)
    bz = gru.bias(res_size, 1)

    Wr_in = nothing
    Wr = nothing
    br = nothing

    return GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br)
end

#in case the user wants to use this driver
function reservoir_driver_params(gru::GRUParams, args...)
    return gru
end

#dispatch on the important function: next_state
function next_state!(out, gru::GRUParams, x, y, W, W_in, b, tmp_array)
    gru_next_state = obtain_gru_state!(out, gru.variant, gru, x, y, W, W_in, b, tmp_array)
    return gru_next_state
end

function allocate_tmp(::GRUParams, tmp_type, res_size)
    return [adapt(tmp_type, zeros(res_size, 1)) for i in 1:9]
end

#W=U, W_in=W in papers. x=h, and y=x. I know, it's confusing. ( on the left our notation)
#fully gated gru
function obtain_gru_state!(out, variant::FullyGated, gru, x, y, W, W_in, b, tmp_array)
    mul!(tmp_array[1], gru.Wz_in, y)
    mul!(tmp_array[2], gru.Wz, x)
    @. tmp_array[3] = gru.activation_function[1](tmp_array[1] + tmp_array[2] + gru.bz)

    mul!(tmp_array[4], gru.Wr_in, y)
    mul!(tmp_array[5], gru.Wr, x)
    @. tmp_array[6] = gru.activation_function[2](tmp_array[4] + tmp_array[5] + gru.br)

    mul!(tmp_array[7], W_in, y)
    mul!(tmp_array[8], W, tmp_array[6] .* x)
    @. tmp_array[9] = gru.activation_function[3](tmp_array[7] + tmp_array[8] + b)
    return @. out = (1 - tmp_array[3]) * x + tmp_array[3] * tmp_array[9]
end

#minimal
function obtain_gru_state!(out, variant::Minimal, gru, x, y, W, W_in, b, tmp_array)
    mul!(tmp_array[1], gru.Wz_in, y)
    mul!(tmp_array[2], gru.Wz, x)
    @. tmp_array[3] = gru.activation_function[1](tmp_array[1] + tmp_array[2] + gru.bz)

    mul!(tmp_array[4], W_in, y)
    mul!(tmp_array[5], W, tmp_array[3] .* x)
    @. tmp_array[6] = gru.activation_function[2](tmp_array[4] + tmp_array[5] + b)

    return @. out = (1 - tmp_array[3]) * x + tmp_array[3] * tmp_array[6]
end
