abstract type LinearModel end

#temp standard ridge
function ESNtrain(esn::AbstractEchoStateNetwork, beta::Float64; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = (train_data*states_new')*inv(add_reg(states_new*states_new', beta))

    return W_out
end

function add_reg(X::AbstractArray{Float64}, beta::Float64)
    n = size(X, 1)
    for i=1:n
        X[i,i] += beta
    end
    return X
end 

#MLJ Ridge
struct Ridge{T<: AbstractFloat} <: LinearModel
    lambda::T
    solver::MLJLinearModels.Solver
end
#Ridge(lambda, solver) = Ridge{Float64}(lambda, solver)
ESNtrain(ridge::Ridge{T}, esn::AbstractEchoStateNetwork; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _ridge(esn, ridge; train_data = esn.train_data)

function _ridge(esn::AbstractEchoStateNetwork, ridge::Ridge; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(train_data, 1), size(states_new, 1))
    for i=1:size(train_data, 1)
        r = RidgeRegression(lambda = ridge.lambda, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(r, states_new', train_data[i,:], solver = ridge.solver)
    end
    
    return W_out
end

#MLJ Lasso
struct Lasso{T<: AbstractFloat} <: LinearModel
    lambda::T
    solver::MLJLinearModels.Solver
end
#Lasso(lambda, solver) = Lasso{Float64}(lambda, solver)
ESNtrain(lasso::Lasso{T}, esn::AbstractEchoStateNetwork; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _lasso(esn, lasso; train_data = esn.train_data)

function _lasso(esn::AbstractEchoStateNetwork, lasso::Lasso; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(train_data, 1), size(states_new, 1))
    for i=1:size(train_data, 1)
        l = LassoRegression(lambda = lasso.lambda, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(l, states_new', train_data[i,:], solver = lasso.solver)
    end
    
    return W_out
end

#MLJ ElastNet 
struct ElastNet{T<: AbstractFloat} <: LinearModel
    lambda::T
    gamma::T
    solver::MLJLinearModels.Solver
end
#ElastNet(lambda, gamma, solver) = ElastNet{Float64}(lambda, gamma, solver)
ESNtrain(elastnet::ElastNet{T}, esn::AbstractEchoStateNetwork; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _elastnet(esn, elastnet; train_data = esn.train_data)

function _elastnet(esn::AbstractEchoStateNetwork, elastnet::ElastNet; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(train_data, 1), size(states_new, 1))
    for i=1:size(train_data, 1)
        en = ElasticNetRegression(lambda = elastnet.lambda, gamma = elastnet.gamma, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(en, states_new', train_data[i,:], solver = elastnet.solver)
    end
    
    return W_out
end

#MLJ Huber -> RobustHuber avoids conflict
struct RobustHuber{T<: AbstractFloat} <: LinearModel
    delta::T
    lambda::T
    gamma::T
    solver::MLJLinearModels.Solver
end
#RobustHuber(delta, lambda, gamma, solver) = RobustHuber{Float64}(delta, lambda, gamma, solver)
ESNtrain(huber::RobustHuber{T}, esn::AbstractEchoStateNetwork; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _huber(esn, huber; train_data = esn.train_data)

function _huber(esn::AbstractEchoStateNetwork, huber::RobustHuber; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(train_data, 1), size(states_new, 1))
    for i=1:size(train_data, 1)
        h = HuberRegression(delta = huber.delta, 
            lambda = huber.lambda, 
            gamma = huber.gamma, 
            fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(h, states_new', train_data[i,:], solver = huber.solver)
    end
    
    return W_out
end
