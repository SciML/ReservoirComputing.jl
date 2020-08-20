abstract type LinearModel end

"""
    ESNtrain(esn::AbstractReservoirComputer, beta::Float64[, train_data])
    
Return the trained output layer using Ridge Regression.
"""
function ESNtrain(esn::AbstractReservoirComputer, beta::Float64; train_data::AbstractArray{Float64} = esn.train_data)
    
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
"""
    Ridge(lambda, solver::MLJLinearModels.Solver)
    
Return a LinearModel object for the training of the model using Ridge Regression with a MLJLinearModels method.
"""
struct Ridge{T<: AbstractFloat} <: LinearModel
    lambda::T
    solver::MLJLinearModels.Solver
end
#Ridge(lambda, solver) = Ridge{Float64}(lambda, solver)
"""
    ESNtrain(lm::LinearModel, esn::AbstractReservoirComputer[, train_data])
    
Return the trained output layer using an MLJLinearModels method built into a LinearSolver struct.
"""
ESNtrain(ridge::Ridge{T}, esn::AbstractReservoirComputer; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _ridge(esn, ridge; train_data = esn.train_data)

function _ridge(esn::AbstractReservoirComputer, ridge::Ridge; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(train_data, 1), size(states_new, 1))
    for i=1:size(train_data, 1)
        r = RidgeRegression(lambda = ridge.lambda, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(r, states_new', train_data[i,:], solver = ridge.solver)
    end
    
    return W_out
end

#MLJ Lasso
"""
    Lasso(lambda, solver::MLJLinearModels.Solver)
    
Return a LinearModel object for the training of the model using Lasso with a MLJLinearModels method.
"""
struct Lasso{T<: AbstractFloat} <: LinearModel
    lambda::T
    solver::MLJLinearModels.Solver
end
#Lasso(lambda, solver) = Lasso{Float64}(lambda, solver)
ESNtrain(lasso::Lasso{T}, esn::AbstractReservoirComputer; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _lasso(esn, lasso; train_data = esn.train_data)

function _lasso(esn::AbstractReservoirComputer, lasso::Lasso; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(train_data, 1), size(states_new, 1))
    for i=1:size(train_data, 1)
        l = LassoRegression(lambda = lasso.lambda, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(l, states_new', train_data[i,:], solver = lasso.solver)
    end
    
    return W_out
end

#MLJ ElastNet 
"""
    ElastNet(lambda, gamma, solver::MLJLinearModels.Solver)
    
Return a LinearModel object for the training of the model using Elastic Net with a MLJLinearModels method.
"""
struct ElastNet{T<: AbstractFloat} <: LinearModel
    lambda::T
    gamma::T
    solver::MLJLinearModels.Solver
end
#ElastNet(lambda, gamma, solver) = ElastNet{Float64}(lambda, gamma, solver)
ESNtrain(elastnet::ElastNet{T}, esn::AbstractReservoirComputer; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _elastnet(esn, elastnet; train_data = esn.train_data)

function _elastnet(esn::AbstractReservoirComputer, elastnet::ElastNet; train_data::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = zeros(Float64, size(train_data, 1), size(states_new, 1))
    for i=1:size(train_data, 1)
        en = ElasticNetRegression(lambda = elastnet.lambda, gamma = elastnet.gamma, fit_intercept = false)
        W_out[i,:] = MLJLinearModels.fit(en, states_new', train_data[i,:], solver = elastnet.solver)
    end
    
    return W_out
end

#MLJ Huber -> RobustHuber avoids conflict
"""
    ElastNet(delta, lambda, gamma, solver::MLJLinearModels.Solver)
    
Return a LinearModel object for the training of the model using the Huber function with a MLJLinearModels method.
"""
struct RobustHuber{T<: AbstractFloat} <: LinearModel
    delta::T
    lambda::T
    gamma::T
    solver::MLJLinearModels.Solver
end
#RobustHuber(delta, lambda, gamma, solver) = RobustHuber{Float64}(delta, lambda, gamma, solver)
ESNtrain(huber::RobustHuber{T}, esn::AbstractReservoirComputer; train_data::AbstractArray{Float64} = esn.train_data) where T<: AbstractFloat = _huber(esn, huber; train_data = esn.train_data)

function _huber(esn::AbstractReservoirComputer, huber::RobustHuber; train_data::AbstractArray{Float64} = esn.train_data)
    
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
