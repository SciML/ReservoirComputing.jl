# Changing Training Algorithms
Notably Echo State Networks have been trained with Ridge Regression algorithms, but the range of useful algorithms to use is much greater. In this section of the documentation it is possible to explore how to use other training methods to obtain the readout layer. All the methods implemented in ReservoirComputing.jl can be used for all models in the library, not only ESNs. The general workflow illustrated in this section will be based on a dummy RC model `my_model = MyModel(...)` that need training in order to obtain the readout layer. The training is done following:
```julia
training_algo = TrainingAlgo()
readout_layer = train(my_model, train_data, training_algo)
```

In this section it is possible to explore how to properly build the `training_algo` and all the possible choices available. In the example section of the documentation it will be provided copy-pastable code to better explore the training algorithms and their impact over the model.

## Linear Models
The library includes a standard implementation of ridge regression, callable using `StandardRidge(regularization_coeff)` where the default value for the regularization coefficent is set to zero. This is also the default model called when no model is specified in `train()`. This makes the default call for traning `train(my_model, train_data)` use Ordinary Least Squares (OLS) for regression.

Leveraging [MLJLinearModels](https://juliaai.github.io/MLJLinearModels.jl/stable/) it is possible to expand the choices of linear models used for the training. The wrappers provided are structured in the following way:
```julia
struct LinearModel
    regression
    solver
    regression_kwargs
end
```
to call the ridge regression using the MLJLinearModels APIs one can use `LinearModel(;regression=LinearRegression)`. It is also possible to use a specific solver, by calling `LinearModel(regression=LinearRegression, solver=Analytical())`. For all the available solvers please reref to the [MLJLinearModels documentation](https://juliaai.github.io/MLJLinearModels.jl/stable/models). To change the regularization coefficient in the ridge example, using for example `lambda = 0.1`, it is needed to pass it in the `regression_kwargs` like so `LinearModel(;regression=LinearRegression, solver=Analytical(), regression_kwargs=(lambda=lambda))`. The nomenclature of the coefficients must follow the MLJLinearModels APIs, using `lambda, gamma` for `LassoRegression` and `delta, lambda, gamma` for `HuberRegression`. Again, please check the [relevant documentation](https://juliaai.github.io/MLJLinearModels.jl/stable/api/) if in doubt. When using MLJLinearModels based regressors do remember to specify `using MLJLinearModels`.

## Gaussian Processes
Another way to obtain the readout layer is possible using Gaussian regression. This is provided through a wrapper of [GaussianProcesses](http://stor-i.github.io/GaussianProcesses.jl/latest/) structured in the following way:
```julia
struct GaussianProcess
    mean
    kernel
    lognoise
    optimize
    optimizer
end
```
While it is necessary to specify a `mean` and a `kernel`, the other defaults are `lognoise=-2, optimize=false, optimizer=Optim.LBFGS()`. For the choice of means and kernels please refer to the proper documentation, [here](http://stor-i.github.io/GaussianProcesses.jl/latest/mean/) and [here](http://stor-i.github.io/GaussianProcesses.jl/latest/kernels/) respectively. 

Building on the simple example given in the GaussianProcesses documentation it is possible to build an intuition of how to use this algorithms for training ReservoirComputing.jl models.
```julia
mZero = MeanZero()   #Zero mean function
kern = SE(0.0,0.0)   #Squared exponential kernel (note that hyperparameters are on the log scale)
logObsNoise = -1.0

gp = GaussianProcess(mZero, kern, lognoise=logObsNoise)
```
Like in the previous case, if one uses GaussianProcesses based regressors it is necessary to specify `using GaussianProcesses`. Additionally, if the optimizer chosen is from an external package, i.e. Optim, that package need to be used in the script as well adding `using Optim`.

## Support Vector Regression
Contrary to the `LinearModel`s and `GaussianProcess`es, no wrappers are needed for support vector regression. By using [LIBSVM.jl](https://github.com/JuliaML/LIBSVM.jl), LIBSVM wrappers in Julia, it is possible to call both `epsilonSVR()` or `nuSVR()` directly in `train()`. For the full range of kernel provided and the parameters to call we refer the user to the official [documentation](https://www.csie.ntu.edu.tw/~cjlin/libsvm/). Like before, if one intends to use LIBSVM regressors it is necessary to specify `using LIBSVM`.
