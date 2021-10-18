
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

function create_states(rnn::RNN, reservoir_matrix, input_matrix, train_data, extended_states)

    train_len = size(train_data, 2)
    res_size = size(reservoir_matrix, 1)
    in_size = size(train_data, 1)
    states = zeros(Float64, res_size, train_len+1) 

    for i=1:train_len-1
        states[:, i+1] = next_state(rnn::RNN, reservoir_matrix, input_matrix, states[:, i], train_data[:, i])

    end

    if extended_states == true
        ext_states = vcat(states, hcat(zeros(Float64, in_size), train_data[:,1:end-1]))
        return ext_states[:,2:end]
    else
        return states[:,2:end]
    end
end

function next_state(rnn::RNN, W, W_in, x, y)
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