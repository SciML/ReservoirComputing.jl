abstract type AbstractLeakyESN <: AbstractEchoStateNetwork end

struct ESN{T, S<:AbstractArray{T}, I, B, F, N} <: AbstractLeakyESN #fix struct
    res_size::I
    in_size::I
    out_size::I
    train_data::S
    alpha::T
    nla_type::N
    activation::F
    reservoir_init::S
    input_layer_init::S
    states::S
    extended_states::B
end

#reservoir matrix W and input layer W_in given by the user
"""
    ESN(W::AbstractArray{T}, train_data::AbstractArray{T}, W_in::AbstractArray{T}
    [, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])

Build an ESN struct given the input and reservoir matrices.
"""
function ESN(res_size, train_data;
             activation = tanh,
             reservoir_init = RandReservoir
             input_layer_init = WeightedInput
             alpha = 1.0,
             nla_type = NLADefault(),
             extended_states = false)

    in_size = size(train_data, 1)
    reservoir = reservoir_init(res_size; kwargs...)#fix kwargs
    input_layer = input_layer_init(res_size, in_size; kwargs...)#fix kwargs

    states = states_matrix(W, W_in, train_data, alpha, activation, extended_states)

    return ESN{T, typeof(train_data),
        typeof(res_size),
        typeof(extended_states),
        typeof(activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states, extended_states)
end


function states_matrix(W::AbstractArray{Float64},
        W_in::AbstractArray{Float64},
        train_data::AbstractArray{Float64},
        alpha::Float64,
        activation::Function,
        extended_states::Bool)

    train_len = size(train_data, 2)
    res_size = size(W, 1)
    in_size = size(train_data, 1)

    states = zeros(Float64, res_size, train_len)
    for i=1:train_len-1
        states[:, i+1] = leaky_fixed_rnn(activation, alpha, W, W_in, states[:, i], train_data[:, i])

    end

    if extended_states == true
        ext_states = vcat(states, hcat(zeros(Float64, in_size), train_data[:,1:end-1]))
        return ext_states
    else
        return states
    end
end

"""
    ESNpredict(esn::AbstractLeakyESN, predict_len::Int, W_out::AbstractArray{Float64})

Return the prediction for a given length of the constructed ESN struct.
"""
function ESNpredict(esn::AbstractLeakyESN,
    predict_len::Int,
    W_out::AbstractArray{Float64})

    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
        end
    elseif esn.extended_states == true
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out)
        end
    end
    return output
end

"""
    ESNpredict_h_steps(esn::AbstractLeakyESN, predict_len::Int, h_steps::Int,
    test_data::AbstractArray{Float64}, W_out::AbstractArray{Float64})

Return the prediction for h steps ahead of the constructed ESN struct.
"""
function ESNpredict_h_steps(esn::AbstractLeakyESN,
    predict_len::Int,
    h_steps::Int,
    test_data::AbstractArray{Float64},
    W_out::AbstractArray{Float64})

    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            if mod(i, h_steps) == 0
                x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, test_data[:,i])
            else
                x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
            end
        end
    elseif esn.extended_states == true
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            if mod(i, h_steps) == 0
                x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], test_data[:,i]), test_data[:,i])
            else
                x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out)
            end
        end
    end
    return output
end

"""
    ESNfitted(esn::AbstractLeakyESN, W_out::Matrix; autonomous=false)

Return the prediction for the training data using the trained output layer. The autonomous trigger can be used to have have it return an autonomous prediction starting from the first point if true, or a point by point prediction if false.
"""

function ESNfitted(esn::AbstractLeakyESN, W_out::Matrix; autonomous=false)
    train_len = size(esn.train_data, 2)
    output = zeros(Float64, esn.in_size, train_len)
    x = zeros(size(esn.states, 1))
    
    if autonomous
        out = esn.train_data[:,1]
        return _fitted!(output, esn, x, train_len, W_out, out)
    else
        return _fitted!(output, esn, x, train_len, W_out, esn.train_data)
    end
end

function _fitted!(output, esn, state, train_len, W_out, vector::Vector)
    if esn.extended_states == false
        for i=1:train_len
            state = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state, vector)
            x_new = nla(esn.nla_type, state)
            vector = (W_out*x_new)
            output[:, i] = vector
        end
    elseif esn.extended_states == true
        for i=1:train_len
            state = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state[1:esn.res_size], vector), vector)
            x_new = nla(esn.nla_type, state)
            vector = (W_out*x_new)
            output[:, i] = vector
        end
    end
    return output
end

function _fitted!(output, esn, state, train_len, W_out, vector::Matrix)
    if esn.extended_states == false
        for i=1:train_len
            state = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state, vector[:,i])
            x_new = nla(esn.nla_type, state)
            out = (W_out*x_new)
            output[:, i] = out
        end
    elseif esn.extended_states == true
        for i=1:train_len
            state = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state[1:esn.res_size], vector[:,i]), vector[:,i])
            x_new = nla(esn.nla_type, state)
            out = (W_out*x_new)
            output[:, i] = out
        end
    end
    return output
end



