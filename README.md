<p align="center">
    <img width="400px" src="docs/src/assets/logo.png"/>
</p>

<div align="center">

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/ReservoirComputing/stable/)
[![arXiv](https://img.shields.io/badge/arXiv-2204.05117-00b300.svg)](https://arxiv.org/abs/2204.05117)
[![codecov](https://codecov.io/gh/SciML/ReservoirComputing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ReservoirComputing.jl)
[![Build Status](https://github.com/SciML/ReservoirComputing.jl/workflows/CI/badge.svg)](https://github.com/SciML/ReservoirComputing.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/db8f91b89a10ad79bbd1d9fdb1340e6f6602a1c0ed9496d4d0.svg)](https://buildkite.com/julialang/reservoircomputing-dot-jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Julia](https://img.shields.io/badge/julia-v1.10+-blue.svg)](https://julialang.org/)
[![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl)

</div>

# ReservoirComputing.jl

ReservoirComputing.jl provides an efficient, modular and easy to use
implementation of Reservoir Computing models such as Echo State Networks (ESNs).
For information on using this package please refer to the
[stable documentation](https://docs.sciml.ai/ReservoirComputing/stable/).
Use the
[in-development documentation](https://docs.sciml.ai/ReservoirComputing/dev/)
to take a look at not yet released features.

## Features

ReservoirComputing.jl provides layers, models, and functions to help build and train
reservoir computing models. More specifically the software offers:

- Base layers for reservoir computing model construction. Main layers provide high level
  reservoir computing building blocks, such as `ReservoirComputer` and `ReservoirChain`.
  Additional, lower level layers provide the building blocks for custom reservoir computers,
  such as `LinearReadout`, `Collect`, `ESNCell`, `DelayLayer`, `NonlinearFeaturesLayer`, and more
- Fully built models:
    + Echo state networks `ESN`
    + Deep echo state networks `DeepESN`
    + Echo state networks with delayed states `DelayESN`
    + Edge of stability echo state networks `ES2N`
    + Euler state networks `EuSN`
    + Hybrid echo state networks `HybridESN`
    + Next generation reservoir computing `NGRC`
- 15+ reservoir initializers and 5+ input layer initializers
- 5+ reservoir states modification algorithms
- Sparse matrix computation through
  [SparseArrays.jl](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
- Multiple training algorithms via [LIBSVM.jl](https://github.com/JuliaML/LIBSVM.jl)
  and [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl)

## Installation

ReservoirComputing.jl can be installed using either of

```julia_repl
julia> ] # press the closing square bracket to enter Pkg mode
pkg> add ReservoirComputing
```
or

```julia
using Pkg
Pkg.add("ReservoirComputing")
```

## Quick Example

To illustrate the workflow of this library we will showcase
how it is possible to train an ESN to learn the dynamics of the
Lorenz system. You can find the same example fully explained in
the [getting started page](https://docs.sciml.ai/ReservoirComputing/stable/getting_started/).

```julia
using OrdinaryDiffEq
using Plots
using Random
using ReservoirComputing

Random.seed!(42)
rng = MersenneTwister(17)

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

prob = ODEProblem(lorenz, [1.0f0, 0.0f0, 0.0f0], (0.0, 200.0), [10.0f0, 28.0f0, 8/3])
data = Array(solve(prob, ABM54(); dt=0.02))
shift = 300
train_len = 5000
predict_len = 1250

input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]

esn = ESN(3, 300, 3; init_reservoir=rand_sparse(; radius=1.2, sparsity=6/300),
    state_modifiers=NLAT2)

ps, st = setup(rng, esn)
ps, st = train!(esn, input_data, target_data, ps, st)
output, st = predict(esn, predict_len, ps, st; initialdata=test[:, 1])

plot(transpose(output)[:, 1], transpose(output)[:, 2], transpose(output)[:, 3];
    label="predicted")
plot!(transpose(test)[:, 1], transpose(test)[:, 2], transpose(test)[:, 3];
    label="actual")
```

## Citing

If you use this library in your work, please cite:

```bibtex
@article{martinuzzi2022reservoircomputing,
  author  = {Francesco Martinuzzi and Chris Rackauckas and Anas Abdelrehim and Miguel D. Mahecha and Karin Mora},
  title   = {ReservoirComputing.jl: An Efficient and Modular Library for Reservoir Computing Models},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {288},
  pages   = {1--8},
  url     = {http://jmlr.org/papers/v23/22-0611.html}
}
```

## Acknowledgements

This project was possible thanks to initial funding through
the [Google summer of code](https://summerofcode.withgoogle.com/)
2020 program. Francesco M. further acknowledges [ScaDS.AI](https://scads.ai/)
and [RSC4Earth](https://rsc4earth.de/) for supporting further progress
on the library. Current developments are possible thanks to research funding
for Francesco M. provided by [MPIPKS](https://www.pks.mpg.de/)
