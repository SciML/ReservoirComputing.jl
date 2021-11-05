function create_states(reservoir_driver::AbstractReservoirDriver, variation, train_data, reservoir_matrix, input_matrix)

    train_len = size(train_data, 2)
    res_size = size(reservoir_matrix, 1)
    in_size = size(train_data, 1)
    states = zeros(res_size, train_len+1) 

    for i=1:train_len
        states[:, i+1] = next_state(reservoir_driver, states[:, i], train_data[:, i], reservoir_matrix, input_matrix)
    end

    states[:,2:end]
end

#standard RNN driver
struct RNN{F,T,R} <: AbstractReservoirDriver
    activation_function::F
    leaky_coefficient::T
    scaling_factor::R
end

function RNN(;activation_function=tanh, leaky_coefficient=1.0, scaling_factor=leaky_coefficient)
    if length(scaling_factor) > 1
        @assert length(activation_function) == length(scaling_factor)
    end
    RNN(activation_function, leaky_coefficient, scaling_factor)
end

function reservoir_driver_params(rnn::RNN, args..)
    rnn
end

function next_state(rnn::RNN, x, y, W, W_in)
    rnn_next_state = (1-rnn.leaky_coefficient).*x
    if length(rnn.scaling_factor) > 1
        for i in rnn.scaling_factor
            rnn_next_state += rnn.scaling_factor[i]*rnn.activation_function[i].((W*x)+(W_in*y))
        end
    else
        rnn_next_state += rnn.scaling_factor*rnn.activation_function.((W*x)+(W_in*y))
    end
    rnn_next_state
end

#GRU-based driver
struct GRU{F,L,V,G}
    activation_function::F
    layer_init::L
    variant::V
end

#layer_init and activation_function must be vectors
function GRU(;activation_function=[sigmoid, sigmoid, tanh], 
              layer_init = fill(DenseLayer(), 6), 
              variant = FullyGated())

    GRU(activation_function, layer_init, variant)
end

#the actual params are only available inside ESN(), so a different driver is needed
struct GRUParams{S} <: AbstractReservoirDriver
    activation_function::F
    variant::V
    U_r::S
    W_r::S
    b_r::S
    U_z::S
    W_z::S
    b_z::S
end

function reservoir_driver_params(gru::GRU, res_size, in_size)
    U_r = create_layer(res_size, in_size, layer_init[1])
    W_r = create_layer(res_size, res_size, layer_init[2])
    b_r = create_layer(res_size, 1, layer_init[3])
    U_z = create_layer(res_size, in_size, layer_init[4])
    W_z = create_layer(res_size, res_size, layer_init[5])
    b_z = create_layer(res_size, 1, layer_init[6])

    GRUParams(activation_function, variant, U_r, W_r, b_r, U_z, W_z, b_z)
end

#in case the user wants to use this driver
function reservoir_driver_params(gru::GRUParams, args...)
    gru
end

#https://arxiv.org/abs/1701.05923# variations of gru

struct FullyGated <: AbstractGRUVariant end
struct Variant1 <: AbstractGRUVariant end
struct Variant2 <: AbstractGRUVariant end
struct Variant3 <: AbstractGRUVariant end
struct Minimal <: AbstractGRUVariant end


"""
    GRUESN(W::AbstractArray{T}, train_data::AbstractArray{T}, W_in::AbstractArray{T} [, gates_weight::T, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])

Return a Gated Recurrent Unit [1] ESN struct.

[1] Cho, Kyunghyun, et al. “Learning phrase representations using RNN encoder-decoder for statistical machine translation.” arXiv preprint arXiv:1406.1078 (2014).
"""

function next_state(gru::GRUParams,x , y, W, W_in)

    gru_next_state = obtain_gru_state(gru.variant, gru::GRUParams, x, y, W, W_in)
end

#dispatch on fully gated gru #check input and output vector
function obtain_gru_state(variant::FullyGated, gru::GRUParams, x, y, W, W_in)
    update_gate = gru.activation_function[3].(gru.U_z*y + gru.W_z*x + gru.b_z)
    reset_gate = gru.activation_function[1].(gru.U_r*y + gru.W_r*x+gru.b_r)
    update = gru.activation_function[2].(W_in*y+W*(reset_gate.*x))
    update_gate.*update + (1 .- update_gate) .*x
end
