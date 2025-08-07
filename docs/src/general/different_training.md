# Changing Training Algorithms

Notably Echo State Networks have been trained with Ridge Regression algorithms, but the range of useful algorithms to use is much greater. In this section of the documentation, it is possible to explore how to use other training methods to obtain the readout layer. All the methods implemented in ReservoirComputing.jl can be used for all models in the library, not only ESNs. The general workflow illustrated in this section will be based on a dummy RC model `my_model = MyModel(...)` that needs training to obtain the readout layer. The training is done as follows:

```julia
training_algo = TrainingAlgo()
readout_layer = train(my_model, train_data, training_algo)
```

In this section, it is possible to explore how to properly build the `training_algo` and all the possible choices available. In the example section of the documentation it will be provided copy-pasteable code to better explore the training algorithms and their impact on the model.

## Linear Models

The library includes a standard implementation of ridge regression, callable using `StandardRidge(regularization_coeff)`. The default regularization coefficient is set to zero. This is also the default model called when no model is specified in `train()`. This makes the default call for training `train(my_model, train_data)` use Ordinary Least Squares (OLS) for regression.

Leveraging [MLJLinearModels](https://juliaai.github.io/MLJLinearModels.jl/stable/) you can expand your choices of linear models for training. The wrappers provided follow this structure:

```julia
struct LinearModel
    regression::Any
    solver::Any
    regression_kwargs::Any
end
```

To call the ridge regression using the MLJLinearModels APIs, you can use `LinearModel(;regression=LinearRegression)`. You can also choose a specific solver by calling, for example, `LinearModel(regression=LinearRegression, solver=Analytical())`. For all the available solvers, please refer to the [MLJLinearModels documentation](https://juliaai.github.io/MLJLinearModels.jl/stable/models/).

To change the regularization coefficient in the ridge example, using for example `lambda = 0.1`, it is needed to pass it in the `regression_kwargs` like so `LinearModel(;regression=LinearRegression, solver=Analytical(), regression_kwargs=(lambda=lambda))`. The nomenclature of the coefficients must follow the MLJLinearModels APIs, using `lambda, gamma` for `LassoRegression` and `delta, lambda, gamma` for `HuberRegression`. Again, please check the [relevant documentation](https://juliaai.github.io/MLJLinearModels.jl/stable/api/) if in doubt. When using MLJLinearModels based regressors, do remember to specify `using MLJLinearModels`.

## Support Vector Regression

Contrary to the `LinearModel`s, no wrappers are needed for support vector regression. By using [LIBSVM.jl](https://github.com/JuliaML/LIBSVM.jl), LIBSVM wrappers in Julia, it is possible to call both `epsilonSVR()` or `nuSVR()` directly in `train()`. For the full range of kernels provided and the parameters to call, we refer the user to the official documentation. Like before, if one intends to use LIBSVM regressors, it is necessary to specify `using LIBSVM`.
