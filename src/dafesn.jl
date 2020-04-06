abstract type AbstractLeakyDAFESN <: AbstractLeakyESN end

struct dafESN{T<:AbstractFloat} <: AbstractLeakyDAFESN
    res_size::Int
    in_size::Int
    out_size::Int
    train_data::Array{T}
    degree::Int
    sigma::T
    alpha::T
    beta::T
    radius::T
    nonlin_alg::Any
    first_activation::Any
    second_activation::Any
    first_lambda::T
    second_lambda::T
    W::AbstractArray{T}
    W_in::AbstractArray{T}
    states::AbstractArray{T}

    function dafESN(approx_res_size::Int,
            train_data::Array{T},
            degree::Int,
            radius::T,
            first_lambda::T,
            second_lambda::T,
            first_activation::Any = tanh,
            second_activation::Any = tanh,
            sigma::T = 0.1,
            alpha::T = 1.0,
            beta::T = 0.0,
            nonlin_alg::Any = NonLinAlgDefault) where T<:AbstractFloat

        in_size = size(train_data)[1]
        out_size = size(train_data)[1] #needs to be different
        res_size = Int(floor(approx_res_size/in_size)*in_size)
        W = init_reservoir(res_size, in_size, radius, degree)
        W_in = init_input_layer(res_size, in_size, sigma)
        states = daf_states_matrix(W, W_in, train_data, alpha, 
        first_activation, second_activation, first_lambda, second_lambda)

        return new{T}(res_size, in_size, out_size, train_data,
        degree, sigma, alpha, beta, radius, nonlin_alg, first_activation, second_activation, first_lambda, second_lambda, W, W_in, states)
    end
end


function daf_states_matrix(W::Matrix{Float64},
        W_in::Matrix{Float64},
        train_data::Array{Float64},
        alpha::Float64,
        first_activation::Function,
        second_activation::Function,
        first_lambda::Float64,
        second_lambda::Float64)

    train_len = size(train_data)[2]
    res_size = size(W)[1]
    states = zeros(Float64, res_size, train_len)
    for i=1:train_len-1
        states[:, i+1] = (1-alpha).*states[:, i] + first_lambda*first_activation.((W*states[:, i])+(W_in*train_data[:, i])) + second_lambda*second_activation.((W*states[:, i])+(W_in*train_data[:, i]))
    end
    return states
end

#function dafESNtrain(esn::dafESN)

#    i_mat = esn.beta.*Matrix(1.0I, esn.res_size, esn.res_size)
#    states_new = esn.nonlin_alg(esn.states)
#    W_out = (esn.train_data*transpose(states_new))*inv(states_new*transpose(states_new)+i_mat)

#    return W_out
#end

function dafESNpredict(esn::AbstractLeakyDAFESN,
    predict_len::Int,
    W_out::Matrix{Float64})

    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]
    for i=1:predict_len
        x_new = esn.nonlin_alg(x)
        out = (W_out*x_new)
        output[:, i] = out
        x = (1-esn.alpha).*x + esn.first_lambda*esn.first_activation.((esn.W*x)+(esn.W_in*out))+ esn.second_lambda*esn.second_activation.((esn.W*x)+(esn.W_in*out))
        
    end
    return output
end
