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

function reservoir_driver_params(rnn::RNN, args...)
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
struct GRU{F,L,R,V} #not an abstractreservoirdriver
    activation_function::F
    layer_init::L
    reservoir_init::R
    variant::V
end

#https://arxiv.org/abs/1701.05923# variations of gru
struct FullyGated <: AbstractGRUVariant end
struct Variant1 <: AbstractGRUVariant end
struct Variant2 <: AbstractGRUVariant end
struct Variant3 <: AbstractGRUVariant end
struct Minimal <: AbstractGRUVariant end

#layer_init and activation_function must be vectors
"""
    GRU(])

Return a Gated Recurrent Unit [1] reservoir driver.

[1] Cho, Kyunghyun, et al. “Learning phrase representations using RNN encoder-decoder for statistical machine translation.” arXiv preprint arXiv:1406.1078 (2014).
"""
function GRU(;activation_function=[NNlib.sigmoid, NNlib.sigmoid, tanh], #has to be a voctor of size 3
              layer_init = fill(DenseLayer(), 5), #has to be a vector of size 5
              reservoir_init = fill(RandSparseReservoir(), 2), #has to be a vector of size 2
              variant = FullyGated())

    GRU(activation_function, layer_init, reservoir_init, variant)
end

#the actual params are only available inside ESN(), so a different driver is needed
struct GRUParams{F,V,S,I,N,SF,IF,NF,T} <: AbstractReservoirDriver
    activation_function::F
    variant::V
    Wz_in::S
    Wz::I
    bz::N
    Wr_in::SF
    Wr::IF
    br::NF
    bh::T
end

#vreation of the actual driver
function reservoir_driver_params(gru::GRU, res_size, in_size)
    gru_params = create_gru_layers(gru, gru.variant, res_size, in_size)
    gru_params
end

#dispatch on the differenct gru variations
function create_gru_layers(gru, variant::FullyGated, res_size, in_size)

    Wz_in = create_layer(res_size, in_size, gru.layer_init[1])
    Wz = create_reservoir(res_size, gru.reservoir_init[1])
    bz = create_layer(res_size, 1, gru.layer_init[2])

    Wr_in = create_layer(res_size, in_size, gru.layer_init[3])
    Wr = create_reservoir(res_size, gru.reservoir_init[2])
    br = create_layer(res_size, 1, gru.layer_init[4])

    bh = create_layer(res_size, 1, gru.layer_init[5])

    GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br, bh)
end

function create_gru_layers(gru, variant::Variant1, res_size, in_size)

    Wz_in = nothing
    Wz = create_reservoir(res_size, gru.reservoir_init[1])
    bz = create_layer(res_size, 1, gru.layer_init[2])

    Wr_in = nothing
    Wr = create_reservoir(res_size, gru.reservoir_init[2])
    br = create_layer(res_size, 1, gru.layer_init[2])

    bh = create_layer(res_size, 1, gru.layer_init[3])

    GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br, bh)
end

function create_gru_layers(gru, variant::Variant2, res_size, in_size)

    Wz_in = nothing
    Wz = create_reservoir(res_size, gru.reservoir_init[1])
    bz = nothing

    Wr_in = nothing
    Wr = create_reservoir(res_size, gru.reservoir_init[2])
    br = nothing

    bh = create_layer(res_size, 1, gru.layer_init[1])

    GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br, bh)
end

function create_gru_layers(gru, variant::Variant3, res_size, in_size)

    Wz_in = nothing
    Wz = nothing
    bz = create_layer(res_size, 1, gru.layer_init[1])

    Wr_in = nothing
    Wr = nothing
    br = create_layer(res_size, 1, gru.layer_init[2])

    bh = create_layer(res_size, 1, gru.layer_init[3])

    GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br, bh)
end

#check this one, not sure
function create_gru_layers(gru, variant::Minimal, res_size, in_size)

    Wz_in = create_layer(res_size, in_size, gru.layer_init[1])
    Wz = create_reservoir(res_size, gru.reservoir_init[1])
    bz = create_layer(res_size, 1, gru.layer_init[2])

    Wr_in = nothing
    Wr = nothing
    br = nothing

    bh = create_layer(res_size, 1, gru.layer_init[3])

    GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br, bh)
end

#in case the user wants to use this driver
function reservoir_driver_params(gru::GRUParams, args...)
    gru
end

#dispatch on the important function: next_state
function next_state(gru::GRUParams,x , y, W, W_in)

    gru_next_state = obtain_gru_state(gru.variant, gru, x, y, W, W_in)
    gru_next_state
end

#W=U, W_in=W in papers. x=h, and y=x. I know, it's confusing. (left our notation)
#fully gated gru
function obtain_gru_state(variant::FullyGated, gru, x, y, W, W_in)

    update_gate = gru.activation_function[1].(gru.Wz_in*y + gru.Wz*x + gru.bz)
    reset_gate = gru.activation_function[2].(gru.Wr_in*y + gru.Wr*x+gru.br)

    update = gru.activation_function[3].(W_in*y+W*(reset_gate.*x)+gru.bh)
    (1 .- update_gate) .*x + update_gate.*update
end

#variant 1
function obtain_gru_state(variant::Variant1, gru, x, y, W, W_in)

    update_gate = gru.activation_function[1].(gru.Wz*x + gru.bz)
    reset_gate = gru.activation_function[2].(gru.Wr*x+gru.br)

    update = gru.activation_function[3].(W_in*y+W*(reset_gate.*x)+gru.bh)
    (1 .- update_gate) .*x + update_gate.*update
end

#variant2
function obtain_gru_state(variant::Variant2, gru, x, y, W, W_in)

    update_gate = gru.activation_function[1].(gru.Wz*x)
    reset_gate = gru.activation_function[2].(gru.Wr*x)

    update = gru.activation_function[3].(W_in*y+W*(reset_gate.*x)+gru.bh)
    (1 .- update_gate) .*x + update_gate.*update
end

#variant 3
function obtain_gru_state(variant::Variant3, gru, x, y, W, W_in)

    update_gate = gru.activation_function[1].(gru.bz)
    reset_gate = gru.activation_function[2].(gru.br)

    update = gru.activation_function[3].(W_in*y+W*(reset_gate.*x)+gru.bh)
    (1 .- update_gate) .*x + update_gate.*update
end


#minimal
function obtain_gru_state(variant::Minimal, gru, x, y, W, W_in)

    forget_gate = gru.activation_function[1].(gru.Wz_in*y + gru.Wz*x + gru.bz)

    update = gru.activation_function[2].(W_in*y+W*(forget_gate.*x)+gru.bh)
    (1 .- forget_gate) .*x + forget_gate.*update
end
