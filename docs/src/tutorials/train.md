# Training Reservoir Computing Models

Training an RC model means fitting the readout. The default objective is ridge
regression; other linear and SVM objectives are available through extensions.

## Ridge regression

```@example training
using ReservoirComputing
using LuxCore: setup
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

## Extending readout fitting

Package extensions can support another training objective by defining:

```julia
ReservoirComputing._fit_readout(
    objective::MyObjective,
    states::AbstractMatrix,
    target_data::AbstractMatrix;
    solver = nothing,
    kwargs...,
)
```

The columns of `states` and `target_data` are aligned training samples. A
matrix-valued result must have size `(n_outputs, n_features)` and is installed
in a standard linear readout. Backends that return another fitted object must
also implement `ReservoirComputing.addreadout!` for compatible model and
readout types.

The model-level [`train`](@ref) function omits the `solver` keyword when the
user supplies `solver=nothing`; otherwise it forwards the solver together with
any additional keywords. Extension methods should reject unsupported solvers
and keywords with an `ArgumentError`.

`_fit_readout` is an internal extension interface and is not a user-facing
training entry point. Its compatibility is not guaranteed before version 1.0.
Users should always call the model-level `train` API.

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
