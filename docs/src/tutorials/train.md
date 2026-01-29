# Training Reservoir Computing Models

Trainig reservoir computing (RC) models usually means solving a linear
regression problem. ReservoirComputing.jl offers multiple stratedies to
provide a readout; in this page we will show the basics, while also pointing out
the possible extensions.

## Training in ReservoirComputing.jl: Ridge Regression

The most simple training of RC models is through ridge regression.
Given the widepread adoption of this training mechnism, ridge regression is the
default training algorithm for RC models in the library.

```@example training
using ReservoirComputing
using Random
Random.seed!(42)
rng = MersenneTwister(42)

input_data = rand(Float32, 3, 100)
target_data = rand(Float32, 5, 100)

model = ESN(3, 100, 5)
ps, st = setup(rng, model)
ps, st = train!(model, input_data, target_data, ps, st,
    StandardRidge(); # default
    solver = QRSolver()) # default
```

In this call you can see that there are two possible knobs to be modified: the
loss function, in this case ridge, and the solver, in this case the build in QR
factorization. In the remainig part of this tutorial we will see how it is possible
to change either.

## Changing Ridge Regression Solver

Building on SciML's [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl), it is
possible to leverage multiple solvers for the ridge problem. For instance, building
on the previous example:

```@example training
using LinearSolve

ps, st = train!(model, input_data, target_data, ps, st,
    StandardRidge(); # default
    solver = SVDFactorization()) # from LinearSolve
```

or 

```@example training
ps, st = train!(model, input_data, target_data, ps, st,
    StandardRidge(); # default
    solver = QRFactorization()) # from LinearSolve
```

For a detailed explanation of the different solvers, as well as a complete list of them,
we suggest visiting the appropriate page in LinearSolve's
[documentation](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/)

## Changing Linear Regression Problem

Linear regression is a general problem, which can be espressed through multiple different
loss functions. While ridge regression is the most common in RC, due to its closed form,
there are multiple other available. ReservoirComputing.jl leverages
[MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) to access all the methods
available from that library.

!!! warn
    
    Currently MLJLinearModels.jl only supports `Float64`. If a certain precision is of the
    upmost importance to you, please refrain from using this external package
    
The train function can be called as before, only this time you can specify different models
and different solvers for the linear regression problem:

```@example training
using MLJLinearModels

ps, st = train!(model, input_data, target_data, ps, st,
    LassoRegression(fit_intercept=false); # from MLJLinearModels
    solver = ProxGrad()) # from MLJLinearModels
```

Make sure to check the MLJLinearModels documentation pages for the available
[models](https://juliaai.github.io/MLJLinearModels.jl/stable/models/) and
[solvers](https://juliaai.github.io/MLJLinearModels.jl/stable/solvers/). Please note that
not all solvers cna be used on all the models. 

!!! note
    
    Currently the support for MLJLinearModels.jl is limited to regressors with
    `fit_intercept=false`. We are working on a solution, but until then you will always
    need to specify it on the regressor.

## Support Vector Regression

ReservoirComputing.jl also allows users to train RC models with support vector regression
through [LIBSVM.jl](https://github.com/JuliaML/LIBSVM.jl). However, the majority of builtin
models in the library uses a [`LinearReadout`](@ref) by default, which can only be trained with
linear regression. In  order to use support vector regression, one needs to build a model
with [`SVMReadout`](@ref)

```@example training
using LIBSVM

model = ReservoirComputer(
    StatefulLayer(ESNCell(3=>100)),
    SVMReadout(100=>5)
)

ps, st = setup(rng, model)
```

We can now train our new `model` similarly to before:

```@example training
ps, st = train!(model, input_data, target_data, ps, st,
    EpsilonSVR() # from LIBSVM
    )
```
