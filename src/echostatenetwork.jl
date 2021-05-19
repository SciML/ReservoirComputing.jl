abstract type AbstractLeakyESN <: AbstractEchoStateNetwork end

struct ESN{T, S<:AbstractArray{T}, I, B, F, N} <: AbstractLeakyESN
    res_size::I
    in_size::I
    out_size::I
    train_data::S
    alpha::T
    nla_type::N
    activation::F
    W::S
    W_in::S
    states::S
    extended_states::B
end

function ESN(approx_res_size::Int,
        train_data::AbstractArray{T},
        degree::Int,
        radius::T;
        activation::Any = tanh,
        sigma::T = 0.1,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = Int(floor(approx_res_size/in_size)*in_size)
    W = init_reservoir_givendeg(res_size, radius, degree)
    W_in = init_input_layer(res_size, in_size, sigma)
    states = states_matrix(W, W_in, train_data, alpha, activation, extended_states)

    return ESN{T, typeof(train_data),
        typeof(res_size),
        typeof(extended_states),
        typeof(activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states, extended_states)
end

#reservoir matrix W given by the user
function ESN(W::AbstractArray{T},
        train_data::Array{T};
        activation::Any = tanh,
        sigma::T = 0.1,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = size(W, 1)
    W_in = init_input_layer(res_size, in_size, sigma)
    states = states_matrix(W, W_in, train_data, alpha, activation, extended_states)

    return ESN{T, typeof(train_data),
        typeof(res_size),
        typeof(extended_states),
        typeof(activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states, extended_states)
end

#input layer W_in given by the user
function ESN(approx_res_size::Int,
        train_data::AbstractArray{T},
        degree::Int,
        radius::T,
        W_in::AbstractArray{T};
        activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1) #needs to be different?
    res_size = Int(floor(approx_res_size/in_size)*in_size)
    W = init_reservoir_givendeg(res_size, radius, degree)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch("size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch("size(W_in, 2) must be equal to in_size"))
    end

    states = states_matrix(W, W_in, train_data, alpha, activation, extended_states)

    return ESN{T, typeof(train_data),
        typeof(res_size),
        typeof(extended_states),
        typeof(activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states, extended_states)
end

#reservoir matrix W and input layer W_in given by the user
"""
    ESN(W::AbstractArray{T}, train_data::AbstractArray{T}, W_in::AbstractArray{T}
    [, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])

Build an ESN struct given the input and reservoir matrices.
"""
function ESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        W_in::AbstractArray{T};
        activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = size(W, 1)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch("size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch("size(W_in, 2) must be equal to in_size"))
    end

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
    x = zeros(esn.res_size)
    
    if autonomous
        out = esn.train_data[:,1]
        return _fitted!(output, esn, x, out)
    else
        return _fitted!(output, esn, x, esn.train_data)
    end
end

function _fitted!(output, esn, state, vector::Vector)
    if esn.extended_states == false
        for i=1:train_len
            state = ReservoirComputing.leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state, vector)
            x_new = nla(esn.nla_type, state)
            vector = (W_out*x_new)
            output[:, i] = vector
        end
    elseif esn.extended_states == true
        for i=1:train_len
            state = ReservoirComputing.leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state, vector)
            x_new = nla(esn.nla_type, state)
            vector = (W_out*x_new)
            output[:, i] = vector
        end
    end
    return output
end

function _fitted!(output, esn, state, vector::Matrix)
    if esn.extended_states == false
        for i=1:train_len
            state = ReservoirComputing.leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state, vector[:,i])
            x_new = nla(esn.nla_type, state)
            out = (W_out*x_new)
            output[:, i] = out
        end
    elseif esn.extended_states == true
        for i=1:train_len
            state = ReservoirComputing.leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, state, vector[:,i])
            x_new = nla(esn.nla_type, state)
            out = (W_out*x_new)
            output[:, i] = out
        end
    end
    return output
end



