# Training Reservoir Computing Models

Training reservoir computing (RC) models usually means solving a linear
regression problem. ReservoirComputing.jl offers multiple strategies to
provide a readout; in this page we will show the basics, while also pointing out
the possible extensions.

## Training in ReservoirComputing.jl: Ridge Regression

The most simple training of RC models is through ridge regression.
Given the widespread adoption of this training mechanism, ridge regression is the
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
ps, st = train(model, input_data, target_data, ps, st;
    objective = StandardRidge(),
    solver = QRFactorization())
```

There are two knobs: the objective (here ridge) and the linear solver (here
LinearSolve's `QRFactorization()`, which is also the package default when
`solver` is omitted).

```@example training
ps, st = train(model, input_data, target_data, ps, st;
    objective = StandardRidge())
```

## Changing Ridge Regression Solver

Other solvers from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) can
be selected explicitly. For example:

```@example training
using LinearSolve

ps, st = train(model, input_data, target_data, ps, st;
    objective = StandardRidge(),
    solver = SVDFactorization())
```

or the legacy built-in path:

```@example training
ps, st = train(model, input_data, target_data, ps, st;
    objective = StandardRidge(),
    solver = QRSolver())
```

For a detailed list of LinearSolve algorithms, see LinearSolve's
[documentation](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/).

`train!` remains available as a compatibility wrapper around `train`.

## Changing Linear Regression Problem

Linear regression is a general problem, which can be expressed through multiple different
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
not all solvers can be used on all the models.

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
