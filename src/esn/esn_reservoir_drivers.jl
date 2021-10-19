function create_states(reservoir_driver::AbstractReservoirDriver, train_data, extended_states, reservoir_matrix, input_matrix)

    train_len = size(train_data, 2)
    res_size = size(reservoir_matrix, 1)
    in_size = size(train_data, 1)
    states = zeros(res_size, train_len+1) 

    for i=1:train_len
        states[:, i+1] = next_state(reservoir_driver, states[:, i], train_data[:, i], reservoir_matrix, input_matrix)
    end
    extended_states ? vcat(states, hcat(zeros(in_size), train_data[:,1:end]))[:,2:end] : states[:,2:end]
end

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