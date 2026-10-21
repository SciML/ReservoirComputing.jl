# Train

```@docs
    train
    train!
    QRSolver
    RidgeRegression
```

## Ridge solvers

For ridge regression, `solver = nothing` selects `QRFactorization()`, the QR
algorithm of [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/).
ReservoirComputing reexports it (see [Reexported API](reexports.md)), so it is
available after `using ReservoirComputing`, and it is the same binding that
LinearSolve exports. Other LinearSolve algorithms, such as `SVDFactorization()`,
require `using LinearSolve`. [`QRSolver`](@ref) is the legacy built-in alternative.
