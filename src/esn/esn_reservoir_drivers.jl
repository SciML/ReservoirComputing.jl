abstract type AbstractReservoirDriver end

"""
    create_states(reservoir_driver::AbstractReservoirDriver, train_data, reservoir_matrix, input_matrix)

Return the trained ESN states according to the given driver.
"""
function create_states(reservoir_driver::AbstractReservoirDriver, train_data, washout, 
    reservoir_matrix, input_matrix, bias_vector)

    train_len = size(train_data, 2)-washout
    res_size = size(reservoir_matrix, 1)
    states = zeros(res_size, train_len)
    _state = zeros(res_size)

    for i=1:washout
        _state = next_state(reservoir_driver, _state, train_data[:, i], reservoir_matrix, input_matrix, bias_vector)
    end

    for j=1:train_len
        _state = next_state(reservoir_driver, _state, train_data[:, washout+j], reservoir_matrix, input_matrix, bias_vector)
        states[:, j] = _state
    end

    states
end

#standard RNN driver
struct RNN{F,T} <: AbstractReservoirDriver
    activation_function::F
    leaky_coefficient::T
end

"""
    RNN(activation_function, leaky_coefficient)
    RNN(;activation_function=tanh, leaky_coefficient=1.0)

Returns a Recurrent Neural Network initializer for the ESN. This is the default choice.
"""
function RNN(;activation_function=tanh, leaky_coefficient=1.0)
    RNN(activation_function, leaky_coefficient)
end

function reservoir_driver_params(rnn::RNN, args...)
    rnn
end

function next_state(rnn::RNN, x, y, W, W_in, b)
    rnn_next_state = (1-rnn.leaky_coefficient).*x
    rnn_next_state += rnn.leaky_coefficient*rnn.activation_function.((W*x).+(W_in*y).+b)
    rnn_next_state
end

#multiple RNN driver
struct MRNN{F,T,R} <: AbstractReservoirDriver
    activation_function::F
    leaky_coefficient::T
    scaling_factor::R
end

"""
    MRNN(activation_function, leaky_coefficient, scaling_factor)
    MRNN(;activation_function=[tanh, sigmoid], leaky_coefficient=1.0, scaling_factor=fill(leaky_coefficient, length(activation_function)))

Returns a Multiple RNN initializer, where multiple function are combined in a linear combination with chosen parameters ```scaling_factor```.
The ```activation_function``` and ```scaling_factor``` arguments must vectors of the same size. Multiple combinations are possible, 
the implementation is based upon a double activation function idea, found in [1].

[1] Lun, Shu-Xian, et al. "_A novel model of leaky integrator echo state network for time-series prediction._" Neurocomputing 159 (2015): 58-66.

"""
function MRNN(;activation_function=[tanh, sigmoid], leaky_coefficient=1.0, scaling_factor=fill(leaky_coefficient, length(activation_function)))
    @assert length(activation_function) == length(scaling_factor)
    MRNN(activation_function, leaky_coefficient, scaling_factor)
end

function reservoir_driver_params(mrnn::MRNN, args...)
    mrnn
end

function next_state(mrnn::MRNN, x, y, W, W_in, b)
    rnn_next_state = (1-mrnn.leaky_coefficient).*x
    for i=1:length(mrnn.scaling_factor)
        rnn_next_state += mrnn.scaling_factor[i]*mrnn.activation_function[i].((W*x).+(W_in*y).+b)
    end
    rnn_next_state
end


#GRU-based driver
struct GRU{F,L,R,V,B} #not an abstractreservoirdriver
    activation_function::F
    inner_layer::L
    reservoir::R
    bias::B
    variant::V
end

#https://arxiv.org/abs/1701.05923# variations of gru
"""
    FullyGated()

Returns a standard Gated Recurrent Unit ESN initializer, as described in [1].

[1] Cho, Kyunghyun, et al. “_Learning phrase representations using RNN encoder-decoder for statistical machine translation._” 
arXiv preprint arXiv:1406.1078 (2014).
"""
struct FullyGated <: AbstractGRUVariant end

"""
    Minimal()

Returns a minimal GRU ESN initializer as described in [1].

[1] Zhou, Guo-Bing, et al. "_Minimal gated unit for recurrent neural networks._" 
International Journal of Automation and Computing 13.3 (2016): 226-234.
"""
struct Minimal <: AbstractGRUVariant end

#layer_init and activation_function must be vectors
"""
    GRU(;activation_function=[NNlib.sigmoid, NNlib.sigmoid, tanh],
        inner_layer = fill(DenseLayer(), 2),
        reservoir = fill(RandSparseReservoir(), 2),
        bias = fill(DenseLayer(), 2),
        variant = FullyGated())

Returns a Gated Recurrent Unit [1] reservoir driver.

[1] Cho, Kyunghyun, et al. “_Learning phrase representations using RNN encoder-decoder for statistical machine translation._” arXiv preprint arXiv:1406.1078 (2014).
"""
function GRU(;activation_function=[NNlib.sigmoid, NNlib.sigmoid, tanh], #has to be a voctor of size 3
              inner_layer = fill(DenseLayer(), 2), #has to be a vector of size 2
              reservoir = fill(RandSparseReservoir(0), 2), #has to be a vector of size 2
              bias = fill(DenseLayer(), 2), #has to be a vector of size 2
              variant = FullyGated())

    GRU(activation_function, inner_layer, reservoir, bias, variant)
end

#the actual params are only available inside ESN(), so a different driver is needed
struct GRUParams{F,V,S,I,N,SF,IF,NF} <: AbstractReservoirDriver
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
    gru_params
end

#dispatch on the differenct gru variations
function create_gru_layers(gru, variant::FullyGated, res_size, in_size)

    Wz_in = create_layer(gru.inner_layer[1], res_size, in_size)
    Wz = create_reservoir(gru.reservoir[1], res_size)
    bz = create_layer(gru.bias[1], res_size, 1)

    Wr_in = create_layer(gru.inner_layer[2], res_size, in_size)
    Wr = create_reservoir(gru.reservoir[2], res_size)
    br = create_layer(gru.bias[2], res_size, 1)

    GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br)
end


#check this one, not sure
function create_gru_layers(gru, variant::Minimal, res_size, in_size)

    Wz_in = create_layer(gru.inner_layer, res_size, in_size)
    Wz = create_reservoir(gru.reservoir, res_size)
    bz = create_layer(gru.bias, res_size, 1)

    Wr_in = nothing
    Wr = nothing
    br = nothing

    GRUParams(gru.activation_function, variant, Wz_in, Wz, bz, Wr_in, Wr, br)
end

#in case the user wants to use this driver
function reservoir_driver_params(gru::GRUParams, args...)
    gru
end

#dispatch on the important function: next_state
function next_state(gru::GRUParams,x , y, W, W_in, b)

    gru_next_state = obtain_gru_state(gru.variant, gru, x, y, W, W_in, b)
    gru_next_state
end

#W=U, W_in=W in papers. x=h, and y=x. I know, it's confusing. (left our notation)
#fully gated gru
function obtain_gru_state(variant::FullyGated, gru, x, y, W, W_in, b)

    update_gate = gru.activation_function[1].(gru.Wz_in*y + gru.Wz*x + gru.bz)
    reset_gate = gru.activation_function[2].(gru.Wr_in*y + gru.Wr*x+gru.br)

    update = gru.activation_function[3].(W_in*y+W*(reset_gate.*x)+b)
    (1 .- update_gate) .*x + update_gate.*update
end

#minimal
function obtain_gru_state(variant::Minimal, gru, x, y, W, W_in, b)

    forget_gate = gru.activation_function[1].(gru.Wz_in*y + gru.Wz*x + gru.bz)

    update = gru.activation_function[2].(W_in*y+W*(forget_gate.*x)+b)
    (1 .- forget_gate) .*x + forget_gate.*update
end
