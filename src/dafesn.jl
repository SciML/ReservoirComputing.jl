abstract type AbstractLeakyDAFESN <: AbstractLeakyESN end

struct dafESN{T<:AbstractFloat} <: AbstractLeakyDAFESN
    res_size::Int
    in_size::Int
    out_size::Int
    train_data::AbstractArray{T}
    alpha::T
    nla_type::NonLinearAlgorithm
    first_activation::Any
    second_activation::Any
    first_lambda::T
    second_lambda::T
    W::AbstractArray{T}
    W_in::AbstractArray{T}
    states::AbstractArray{T}
    extended_states::Bool
end


function dafESN(approx_res_size::Int,
        train_data::AbstractArray{T},
        degree::Int,
        radius::T,
        first_lambda::T,
        second_lambda::T,
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

    return dafESN{T}(res_size, in_size, out_size, train_data,
    alpha, nla_type, first_activation, second_activation, first_lambda,
    second_lambda, W, W_in, states, extended_states)
end

#reservoir matrix W given by the user
function dafESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        first_lambda::T,
        second_lambda::T,
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

    return dafESN{T}(res_size, in_size, out_size, train_data,
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
        W_in::AbstractArray{T},
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

    return dafESN{T}(res_size, in_size, out_size, train_data,
    alpha, nla_type, first_activation, second_activation, first_lambda,
    second_lambda, W, W_in, states, extended_states)
end

#reservoir matrix W and input layer W_in given by the user
function dafESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        first_lambda::T,
        second_lambda::T,
        W_in::AbstractArray{T},
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

    return dafESN{T}(res_size, in_size, out_size, train_data,
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

    train_len = size(train_data)[2]
    res_size = size(W)[1]
    states = zeros(Float64, res_size, train_len)
    for i=1:train_len-1
        states[:, i+1] = (1-alpha).*states[:, i] + first_lambda*first_activation.((W*states[:, i])+(W_in*train_data[:, i])) + second_lambda*second_activation.((W*states[:, i])+(W_in*train_data[:, i]))
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
            x = (1-esn.alpha).*x + esn.first_lambda*esn.first_activation.((esn.W*x)+(esn.W_in*out))+
            esn.second_lambda*esn.second_activation.((esn.W*x)+(esn.W_in*out))
        end
    else
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            x = vcat((1-esn.alpha).*x[1:esn.res_size] + esn.first_lambda*esn.first_activation.((esn.W*x[1:esn.res_size])+(esn.W_in*out))+
            esn.second_lambda*esn.second_activation.((esn.W*x[1:esn.res_size])+(esn.W_in*out)), out)
        end
    end

    return output
end
