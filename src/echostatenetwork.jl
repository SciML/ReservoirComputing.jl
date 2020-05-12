abstract type AbstractLeakyESN <: AbstractEchoStateNetwork end

struct ESN{T<:AbstractFloat} <: AbstractLeakyESN
    res_size::Int
    in_size::Int
    out_size::Int
    train_data::AbstractArray{T}
    #degree::Int
    #sigma::T
    alpha::T
    #radius::T
    nla_type::NonLinearAlgorithm
    activation::Any
    W::AbstractArray{T}
    W_in::AbstractArray{T}
    states::AbstractArray{T}

end

function ESN(approx_res_size::Int,
        train_data::AbstractArray{T},
        degree::Int,
        radius::T,
        activation::Any = tanh,
        sigma::T = 0.1,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault()) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1) #needs to be different?
    res_size = Int(floor(approx_res_size/in_size)*in_size)
    W = init_reservoir(res_size, in_size, radius, degree)
    W_in = init_input_layer(res_size, in_size, sigma)
    states = states_matrix(W, W_in, train_data, alpha, activation)

    return ESN{T}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states)
end

#reservoir matrix W given by the user
function ESN(W::AbstractArray{T},
        train_data::Array{T},
        activation::Any = tanh,
        sigma::T = 0.1,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault()) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1) 
    res_size = size(W, 1)
    W_in = init_input_layer(res_size, in_size, sigma)
    states = states_matrix(W, W_in, train_data, alpha, activation)

    return ESN{T}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states)
end

#input layer W_in given by the user
function ESN(approx_res_size::Int,
        train_data::AbstractArray{T},
        degree::Int,
        radius::T,
        W_in::AbstractArray{T},
        activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault()) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1) #needs to be different?
    res_size = Int(floor(approx_res_size/in_size)*in_size)
    W = init_reservoir(res_size, in_size, radius, degree)
    
    if size(W_in, 1) != res_size
        throw(DimensionMismatch(W_in, "size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch(W_in, "size(W_in, 2) must be equal to in_size"))
    end
    
    states = states_matrix(W, W_in, train_data, alpha, activation)

    return ESN{T}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states)
end

#reservoir matrix W and input layer W_in given by the user
function ESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        W_in::AbstractArray{T},
        activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault()) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1) 
    res_size = size(W, 1)
    
    if size(W_in, 1) != res_size
        throw(DimensionMismatch(W_in, "size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch(W_in, "size(W_in, 2) must be equal to in_size"))
    end    
    
    states = states_matrix(W, W_in, train_data, alpha, activation)

    return ESN{T}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, states)
end

function init_reservoir(res_size::Int,
        in_size::Int,
        radius::Float64,
        degree::Int)

    sparsity = degree/res_size
    W = Matrix(sprand(Float64, res_size, res_size, sparsity))
    W = 2.0 .*(W.-0.5)
    replace!(W, -1.0=>0.0)
    rho_w = maximum(abs.(eigvals(W)))
    W .*= radius/rho_w
    return W
end

function init_input_layer(res_size::Int,
        in_size::Int,
        sigma::Float64)

    W_in = zeros(Float64, res_size, in_size)
    q = Int(res_size/in_size)
    for i=1:in_size
        W_in[(i-1)*q+1 : (i)*q, i] = (2*sigma).*(rand(Float64, 1, q).-0.5)
    end
    return W_in
end

function states_matrix(W::AbstractArray{Float64},
        W_in::AbstractArray{Float64},
        train_data::AbstractArray{Float64},
        alpha::Float64,
        activation::Function)

    train_len = size(train_data)[2]
    res_size = size(W)[1]
    states = zeros(Float64, res_size, train_len)
    for i=1:train_len-1
        states[:, i+1] = (1-alpha).*states[:, i] + alpha*activation.((W*states[:, i])+(W_in*train_data[:, i]))
    end
    return states
end


function ESNpredict(esn::AbstractLeakyESN,
    predict_len::Int,
    W_out::AbstractArray{Float64})

    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]
    for i=1:predict_len
        x_new = nla(esn.nla_type, x)
        out = (W_out*x_new)
        output[:, i] = out
        x = (1-esn.alpha).*x + esn.alpha*esn.activation.((esn.W*x)+(esn.W_in*out))
    end
    return output
end


#needs better implementation
function ESNsingle_predict(esn::AbstractLeakyESN,
    predict_len::Int,
    partial::AbstractArray{Float64},
    test_data::AbstractArray{Float64},
    W_out::AbstractArray{Float64})

    output = zeros(Float64, esn.in_size, predict_len)
    out_new = zeros(Float64, esn.out_size)
    x = esn.states[:, end]
    for i=1:predict_len
        x_new = nla(esn.nla_type, x)
        output[:, i] = out_new        
        x = (1-esn.alpha).*x + esn.alpha*esn.activation.((esn.W*x)+(esn.W_in*out_new))
    end
    return output
end

