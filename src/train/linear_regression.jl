
"""
    ESNtrain(esn::AbstractReservoirComputer, beta::Float64[, train_data])

Return the trained output layer using Ridge Regression.
"""

struct StandardRidge{T} <: AbstractLinearModel
    regularization_coeff::T
end

function StandardRidge(;regularization_coeff=reg_coeff)
    StandardRidge(regularization_coeff)
end

#default training - OLS
function train(rc::AbstractReservoirComputer, target_data, sr::StandardRidge=StandardRidge(0.0))
    states_new = nla(rc.nla_type, rc.states)
    (target_data*states_new')*inv(add_reg(states_new*states_new', sr.regularization_coeff))    
end

function add_reg(X, beta)
    n = size(X, 1)
    for i=1:n
        X[i,i] += beta
    end
    return X
end

#mlj interface
struct LinearModel{T,S,K} <: AbstractLinearModel
    regression::T
    solver::S
    regression_kwargs::K
end

function LinearModel(;regression=LinearRegression, 
                 solver=Analytical(), 
                 regression_kwargs=(;))
                 LinearModel(regression, solver, regression_kwargs)
end

function train(rc::AbstractReservoirComputer, target_data, linear::LinearModel)

    states_new = nla(rc.nla_type, rc.states)
    output_layer = zeros(size(target_data, 1), size(states_new, 1))
    for i=1:size(target_data, 1)
        regressor = linear.regression(; fit_intercept = false, linear.regression_kwargs...)
        output_layer[i,:] = MLJLinearModels.fit(regressor, states_new', 
        target_data[i,:], solver = linear.solver)
    end
    output_layer
end
