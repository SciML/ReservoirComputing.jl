abstract type LinearModel end


function ESNtrain(esn::AbstractEchoStateNetwork, beta::Float64)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = (esn.train_data*states_new')*inv(add_reg(states_new*states_new', beta))

    return W_out
end

function add_reg(X::AbstractArray{Float64}, beta::Float64)
    n = size(X, 1)
    for i=1:n
        X[i,i] += beta
    end
    return X
end 

struct Ridge{T<: AbstractFloat} <: LinearModel
    beta::T
end
Ridge(beta) = Ridge{Float64}(beta)
ESNtrain(ridge::Ridge{T}, esn::AbstractEchoStateNetwork) where T<: AbstractFloat = _ridge(esn, ridge)

function _ridge(esn::AbstractEchoStateNetwork, ridge::Ridge)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(esn.train_data, 1), size(states_new, 1))
    for i=1:size(esn.train_data, 1)
        r = RidgeRegression(lambda = ridge.beta, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(r, states_new', esn.train_data[i,:])
    end
    
    return W_out
end

struct Lasso{T<: AbstractFloat} <: LinearModel
    beta::T
    max_iter::Int
end
Lasso(beta, max_iter) = Lasso{Float64}(beta, max_iter)
ESNtrain(lasso::Lasso{T}, esn::AbstractEchoStateNetwork) where T<: AbstractFloat = _lasso(esn, lasso)

function _lasso(esn::AbstractEchoStateNetwork, lasso::Lasso)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(esn.train_data, 1), size(states_new, 1))
    for i=1:size(esn.train_data, 1)
        l = LassoRegression(lambda = lasso.beta, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(l, states_new', esn.train_data[i,:], solver = ProxGrad(max_iter = lasso.max_iter))
    end
    
    return W_out
end

struct ElastNet{T<: AbstractFloat} <: LinearModel
    beta::T
    beta2::T
    max_iter::Int
end
ElastNet(beta, beta2, max_iter) = ElastNet{Float64}(beta, beta2, max_iter)
ESNtrain(elastnet::ElastNet{T}, esn::AbstractEchoStateNetwork) where T<: AbstractFloat = _elastnet(esn, elastnet)

function _elastnet(esn::AbstractEchoStateNetwork, elastnet::ElastNet)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(esn.train_data, 1), size(states_new, 1))
    for i=1:size(esn.train_data, 1)
        en = ElasticNetRegression(lambda = elastnet.beta, gamma = elastnet.beta2, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(en, states_new', esn.train_data[i,:], solver = ProxGrad(max_iter = elastnet.max_iter))
    end
    
    return W_out
end
