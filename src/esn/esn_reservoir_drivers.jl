
struct RNN{F,T,R} <: AbstractReservoirDriver
    activation_function::F
    leaky_coefficient::T
    scaling_factor::R
end

function RNN(;activation_function=tanh, leaky_coefficient=1.0, scaling_factor=leaky_coefficient)
    @assert length(activation_function) == length(scaling_factor)
    RNN(activation_function, leaky_coefficient, scaling_factor)
end

function create_states(reservoir_matrix, input_matrix, train_data, extended_states, nla_type, reservoir_driver::RNN)
    train_len = size(train_data, 2)
    res_size = size(reservoir_matrix, 1)
    in_size = size(train_data, 1)

    states = zeros(Float64, res_size, train_len)
    for i=1:train_len-1
        states[:, i+1] = leaky_fixed_rnn(reservoir_driver.activation_function, reservoir_driver.leaky_coefficient, reservoir_driver.scaling_factor, 
                                         reservoir_matrix, input_matrix, states[:, i], train_data[:, i])

    end

    if extended_states == true
        ext_states = vcat(states, hcat(zeros(Float64, in_size), train_data[:,1:end-1]))
        return nla(nla_type, ext_states)
    else
        return nla(nla_type, states)
    end
end

function leaky_fixed_rnn(activation, alpha, scaling, W, W_in, x, y)
    rnn_out = (1-alpha).*x
    for i in scaling
        rnn_out += scaling[i]*activation[i].((W*x)+(W_in*y))
    end
    rnn_out
end