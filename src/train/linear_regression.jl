
"""
    ESNtrain(esn::AbstractReservoirComputer, beta::Float64[, train_data])

Return the trained output layer using Ridge Regression.
"""

struct StandardRidge{T} <: LinearModel
    regularization_coeff::T
end

function StandardRidge(;regularization_coeff=reg_coeff)
    StandardRidge(regularization_coeff)
end

#default training - OLS

function train!(esn::AbstractReservoirComputer, target_data, sr::StandardRidge=StandardRidge(0.0))
    states_new = nla(esn.nla_type, esn.states)
    esn.output_layer = (target_data*states_new')*inv(add_reg(states_new*states_new', sr.regularization_coeff))    
end

function add_reg(X, beta)
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
struct Ridge{T,I} <: LinearModel
    lambda::T
    solver::MLJLinearModels.Solver
    ridge_kwargs::I
end
#Ridge(lambda, solver) = Ridge{Float64}(lambda, solver)

function Ridge(lambda_arg; 
               lambda=lambda_arg, 
               solver=Analytical(), 
               ridge_kwargs=(;fit_intercept = false))
    
    Ridge(lambda, solver, ridge_kwargs)
end

function train!(esn::AbstractReservoirComputer, target_data, ridge::Ridge)

    states_new = nla(esn.nla_type, esn.states)
    esn.output_layer = zeros(size(target_data, 1), size(states_new, 1))
    for i=1:size(target_data, 1)
        r = RidgeRegression(ridge.lambda; ridge.ridge_kwargs...)
        esn.output_layer[i,:] = MLJLinearModels.fit(r, states_new', target_data[i,:], solver = ridge.solver)
    end
end

#MLJ Lasso
"""
    Lasso(lambda, solver::MLJLinearModels.Solver)

Return a LinearModel object for the training of the model using Lasso with a MLJLinearModels method.
"""
struct Lasso{T,I} <: LinearModel
    lambda::T
    solver::MLJLinearModels.Solver
    lasso_kwargs::I
end

function Lasso(lambda_arg; 
               lambda=lambda_arg, 
               solver=ProxGrad(), 
               lasso_kwargs=(;fit_intercept = false))
    Lasso(lambda, solver, lasso_kwargs)
end

function train!(esn::AbstractReservoirComputer, target_data, lasso::Lasso)

    states_new = nla(esn.nla_type, esn.states)
    esn.output_layer = zeros(size(target_data, 1), size(states_new, 1))
    for i=1:size(target_data, 1)
        l = LassoRegression(lasso.lambda; lasso.lasso_kwargs...)
        esn.output_layer[i,:] = MLJLinearModels.fit(l, states_new', target_data[i,:], solver = lasso.solver)
    end
end

#MLJ ElastNet
"""
    ElastNet(lambda, gamma, solver::MLJLinearModels.Solver)

Return a LinearModel object for the training of the model using Elastic Net with a MLJLinearModels method.
"""
struct ElastNet{T,I} <: LinearModel
    lambda::T
    gamma::T
    solver::MLJLinearModels.Solver
    elastnet_kwargs::I
end

function ElastNet(lambda_arg, gamma_arg; 
                  lambda=lambda_arg,
                  gamma=gamma_arg,
                  solver=ProxGrad(),
                  elastnet_kwargs=(;fit_intercept = false))
    ElastNet(lambda, gamma, solver, elastnet_kwargs)
end

function train!(esn::AbstractReservoirComputer, target_data, elastnet::ElastNet)

    states_new = nla(esn.nla_type, esn.states)
    esn.output_layer = zeros(size(target_data, 1), size(states_new, 1))
    for i=1:size(target_data, 1)
        en = ElasticNetRegression(elastnet.lambda,  elastnet.gamma; elastnet.elastnet_kwargs...)
        esn.output_layer[i,:] = MLJLinearModels.fit(en, states_new', target_data[i,:], solver = elastnet.solver)
    end
end

#MLJ Huber -> RobustHuber avoids conflict
"""
    ElastNet(delta, lambda, gamma, solver::MLJLinearModels.Solver)

Return a LinearModel object for the training of the model using the Huber function with a MLJLinearModels method.
"""
struct RobustHuber{T,I} <: LinearModel
    delta::T
    lambda::T
    gamma::T
    solver::MLJLinearModels.Solver
    huber_kwargs::I
end

function RobustHuber(delta_arg, lambda_arg, gamma_arg;
            delta=delta_arg,
            lambda=lambda_arg,
            gamma=gamma_arg,
            solver=MLJLinearModels.Newton(),
            huber_kwargs=(;fit_intercept = false))
    RobustHuber(delta, lambda, gamma, solver, huber_kwargs)
end

function train!(esn::AbstractReservoirComputer, target_data, huber::RobustHuber)

    states_new = nla(esn.nla_type, esn.states)
    esn.output_layer = zeros(Float64, size(target_data, 1), size(states_new, 1))
    for i=1:size(target_data, 1)
        h = HuberRegression(huber.delta, huber.lambda, huber.gamma; huber.huber_kwargs...)
        esn.output_layer[i,:] = MLJLinearModels.fit(h, states_new', target_data[i,:], solver = huber.solver)
    end
end

#states_new = vcat(states_new, hesn.physics_model_data[:, 2:end])