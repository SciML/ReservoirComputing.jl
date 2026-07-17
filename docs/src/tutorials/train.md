# Training Reservoir Computing Models

Training an RC model means fitting the readout. The default objective is ridge
regression; other linear and SVM objectives are available through extensions.

## Ridge regression

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
    objective = RidgeRegression(),
    solver = QRFactorization())
```

`objective` chooses what to fit (here ridge). `solver` chooses how to solve it;
omitting `solver` uses [`QRFactorization`](@ref).

```@example training
ps, st = train(model, input_data, target_data, ps, st;
    objective = RidgeRegression())
```

## Changing the ridge solver

Other [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) algorithms:

```@example training
using LinearSolve

ps, st = train(model, input_data, target_data, ps, st;
    objective = RidgeRegression(),
    solver = SVDFactorization())
```

Legacy built-in path:

```@example training
ps, st = train(model, input_data, target_data, ps, st;
    objective = RidgeRegression(),
    solver = QRSolver())
```

See LinearSolve's
[solver list](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/).

## Other linear objectives

[MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) provides
additional regressors (lasso, elastic net, …).

!!! warn

    MLJLinearModels currently supports `Float64` only.

```@example training
using MLJLinearModels

ps, st = train(model, input_data, target_data, ps, st;
    objective = LassoRegression(fit_intercept = false),
    solver = ProxGrad())
```

See MLJLinearModels
[models](https://juliaai.github.io/MLJLinearModels.jl/stable/models/) and
[solvers](https://juliaai.github.io/MLJLinearModels.jl/stable/solvers/). Not
every solver works with every model. MLJ also exports a type named
`RidgeRegression`; write `MLJLinearModels.RidgeRegression` when both packages
are loaded.

!!! note

    Only regressors with `fit_intercept=false` are supported for now.

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
ps, st = train(model, input_data, target_data, ps, st;
    objective = EpsilonSVR()) # from LIBSVM
```
