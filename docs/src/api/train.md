# Train

```@docs
    train
    train!
    QRSolver
    QRFactorization
```

For ridge regression, `solver = nothing` selects LinearSolve's
`QRFactorization()` (re-exported by ReservoirComputing). Other LinearSolve
algorithms require `using LinearSolve`.

## Training methods

```@docs
    StandardRidge
```
