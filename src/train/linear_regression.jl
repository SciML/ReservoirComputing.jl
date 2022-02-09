struct StandardRidge{T} <: AbstractLinearModel
    regularization_coeff::T
end

"""
    StandardRidge(regularization_coeff)
    StandardRidge(;regularization_coeff=0.0)

Ridge regression training for all the models in the library. The ```regularization_coeff``` is the regularization, 
it can be passed as an arg or kwarg.
"""
function StandardRidge(;regularization_coeff=0.0)
    StandardRidge(regularization_coeff)
end

#default training - OLS
function _train(states, target_data, sr::StandardRidge=StandardRidge(0.0))
    out_size = size(target_data, 1)
    output_layer = (target_data*states')*inv(add_reg(states*states', sr.regularization_coeff))
    OutputLayer(sr, output_layer, out_size, target_data[:,end])
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

"""
    LinearModel(;regression=LinearRegression, 
        solver=Analytical(), 
        regression_kwargs=(;))

Linear regression training based on [MLJLinearModels](https://juliaai.github.io/MLJLinearModels.jl/stable/) for all the models in the library. 
All the parameters have to be passed into ```regression_kwargs```, apart from the solver choice. MLJLinearModels.jl needs to be called in order 
to use these models.
"""
function LinearModel(;regression=LinearRegression, 
                 solver=Analytical(), 
                 regression_kwargs=(;))
    LinearModel(regression, solver, regression_kwargs)
end

function LinearModel(regression; 
    solver=Analytical(), 
    regression_kwargs=(;))
LinearModel(regression, solver, regression_kwargs)
end

function _train(states, target_data, linear::LinearModel)

    out_size = size(target_data, 1)
    output_layer = zeros(size(target_data, 1), size(states, 1))
    for i=1:size(target_data, 1)
        regressor = linear.regression(; fit_intercept = false, linear.regression_kwargs...)
        output_layer[i,:] = MLJLinearModels.fit(regressor, states', 
        target_data[i,:], solver = linear.solver)
    end
    OutputLayer(linear, output_layer, out_size, target_data[:,end])
end
