abstract type AbstractLeakyDAFESN <: AbstractLeakyESN end

struct dafESN{T, S<:AbstractArray{T}, I, B, F1, F2, N} <: AbstractLeakyDAFESN
    res_size::I
    in_size::I
    out_size::I
    train_data::S
    alpha::T
    nla_type::N
    first_activation::F1
    second_activation::F2
    first_lambda::T
    second_lambda::T
    W::S
    W_in::S
    states::S
    extended_states::B
end


function dafESN(approx_res_size::Int,
        train_data::AbstractArray{T},
        degree::Int,
        radius::T,
        first_lambda::T,
        second_lambda::T;
        first_activation::Any = tanh,
        second_activation::Any = tanh,
        sigma::T = 0.1,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = Int(floor(approx_res_size/in_size)*in_size)

    W = init_reservoir_givendeg(res_size, radius, degree)
    W_in = init_input_layer(res_size, in_size, sigma)
    states = daf_states_matrix(W, W_in, train_data, alpha,
    first_activation, second_activation, first_lambda, second_lambda, extended_states)

    return dafESN{T, typeof(train_data), 
        typeof(res_size), 
        typeof(extended_states), 
        typeof(first_activation), 
        typeof(second_activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, first_activation, second_activation, first_lambda,
    second_lambda, W, W_in, states, extended_states)
end

#reservoir matrix W given by the user
function dafESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        first_lambda::T,
        second_lambda::T;
        first_activation::Any = tanh,
        second_activation::Any = tanh,
        sigma::T = 0.1,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = size(W, 1)

    W_in = init_input_layer(res_size, in_size, sigma)
    states = daf_states_matrix(W, W_in, train_data, alpha,
    first_activation, second_activation, first_lambda, second_lambda, extended_states)

    return dafESN{T, typeof(train_data), 
        typeof(res_size), 
        typeof(extended_states), 
        typeof(first_activation), 
        typeof(second_activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, first_activation, second_activation, first_lambda,
    second_lambda, W, W_in, states, extended_states)
end


#input layer W_in given by the user
function dafESN(approx_res_size::Int,
        train_data::AbstractArray{T},
        degree::Int,
        radius::T,
        first_lambda::T,
        second_lambda::T,
        W_in::AbstractArray{T};
        first_activation::Any = tanh,
        second_activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = Int(floor(approx_res_size/in_size)*in_size)
    W = init_reservoir_givendeg(res_size, radius, degree)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch(W_in, "size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch(W_in, "size(W_in, 2) must be equal to in_size"))
    end

    states = daf_states_matrix(W, W_in, train_data, alpha,
    first_activation, second_activation, first_lambda, second_lambda, extended_states)

    return dafESN{T, typeof(train_data), 
        typeof(res_size), 
        typeof(extended_states), 
        typeof(first_activation), 
        typeof(second_activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, first_activation, second_activation, first_lambda,
    second_lambda, W, W_in, states, extended_states)
end

#reservoir matrix W and input layer W_in given by the user
function dafESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        first_lambda::T,
        second_lambda::T,
        W_in::AbstractArray{T};
        first_activation::Any = tanh,
        second_activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = size(W, 1)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch(W_in, "size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch(W_in, "size(W_in, 2) must be equal to in_size"))
    end

    states = daf_states_matrix(W, W_in, train_data, alpha,
    first_activation, second_activation, first_lambda, second_lambda, extended_states)

    return dafESN{T, typeof(train_data), 
        typeof(res_size), 
        typeof(extended_states), 
        typeof(first_activation), 
        typeof(second_activation),
        typeof(nla_type)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, first_activation, second_activation, first_lambda,
    second_lambda, W, W_in, states, extended_states)
end

function daf_states_matrix(W::AbstractArray{Float64},
        W_in::AbstractArray{Float64},
        train_data::AbstractArray{Float64},
        alpha::Float64,
        first_activation::Function,
        second_activation::Function,
        first_lambda::Float64,
        second_lambda::Float64,
        extended_states::Bool)

    train_len = size(train_data, 2)
    res_size = size(W, 1)
    in_size = size(train_data, 1)
    states = zeros(Float64, res_size, train_len)
    for i=1:train_len-1
        states[:, i+1] = double_leaky_fixed_rnn(alpha, first_activation, second_activation, first_lambda, second_lambda, W, W_in, states[:, i], train_data[:, i])
    end

    if extended_states == true
        ext_states = vcat(states, hcat(zeros(Float64, in_size), train_data[:,1:end-1]))
        return ext_states
    else
        return states
    end

    return states
end



function dafESNpredict(esn::AbstractLeakyDAFESN,
    predict_len::Int,
    W_out::AbstractArray{Float64})

    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            x = double_leaky_fixed_rnn(esn.alpha, esn.first_activation, esn.second_activation, esn.first_lambda, esn.second_lambda, esn.W, esn.W_in, x, out)
        end
    elseif esn.extended_states == true
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            x = vcat(double_leaky_fixed_rnn(esn.alpha, esn.first_activation, esn.second_activation, esn.first_lambda, esn.second_lambda, esn.W, esn.W_in, x[1:esn.res_size], out), out)
        end
    end

    return output
end

function dafESNpredict_h_steps(esn::AbstractLeakyESN,
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
                x = double_leaky_fixed_rnn(esn.alpha, esn.first_activation, esn.second_activation, esn.first_lambda, esn.second_lambda, esn.W, esn.W_in, x, test_data[:,i])
            else
                x = double_leaky_fixed_rnn(esn.alpha, esn.first_activation, esn.second_activation, esn.first_lambda, esn.second_lambda, esn.W, esn.W_in, x, out)
            end
        end
    elseif esn.extended_states == true
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            if mod(i, h_steps) == 0
                x = vcat(double_leaky_fixed_rnn(esn.alpha, esn.first_activation, esn.second_activation, esn.first_lambda, esn.second_lambda, esn.W, esn.W_in, x[1:esn.res_size], test_data[:,i]), test_data[:,i])
            else
                x = vcat(double_leaky_fixed_rnn(esn.alpha, esn.first_activation, esn.second_activation, esn.first_lambda, esn.second_lambda, esn.W, esn.W_in, x[1:esn.res_size], out), out)
            end
        end
    end
    return output
end  
