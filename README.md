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

ReservoirComputing.jl provides layers,models, and functions to help build and train
reservoir computing models. More specifically the software offers

- Base layers for reservoir computing model construction such as `ReservoirChain`,
  `Readout`, `Collect`, and `ESNCell`
- Fully built models such as `ESN`, and `DeepESN`
- 15+ reservoir initializers and 5+ input layer initializers
- 5+ reservoir states modification algorithms
- Sparse matrix computation through
  [SparseArrays.jl](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)

## Installation

ReservoirComputing.jl can be installed using either of

```julia_repl
julia> ] #actually press the closing square brackets
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
Lorenz system.

### 1. Generate data

As a general first step wee fix the random seed for reproducibilty

```julia
using Random
Random.seed!(42)
rng = MersenneTwister(17)
```

For an autoregressive prediction we need the target data
to be one step ahead of the training data:

```julia
using OrdinaryDiffEq

#define lorenz system
function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

#solve and take data
prob = ODEProblem(lorenz, [1.0f0, 0.0f0, 0.0f0], (0.0, 200.0), [10.0f0, 28.0f0, 8/3])
data = Array(solve(prob, ABM54(); dt=0.02))
shift = 300
train_len = 5000
predict_len = 1250

#one step ahead for generative prediction
input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
```

### 2. Build Echo State Network

We can either use the provided `ESN` or build one from scratch.
We showcase the first option:

```julia
using ReservoirComputing
input_size = 3
res_size = 300
esn = ESN(input_size, res_size, input_size;
    init_reservoir=rand_sparse(; radius=1.2, sparsity=6/300),
    state_modifiers=NLAT2
)
```

### 3. Train the Echo State Network

ReservoirComputing.jl builds on Lux(Core), so in order to train the model
we first need to instantiate the parameters and the states:

```julia
ps, st = setup(rng, esn)
ps, st = train!(esn, input_data, target_data, ps, st)
```

### 4. Predict and visualize

We can now use the trained ESN to forecast the Lorenz system dynamics

```julia
output, st = predict(esn, 1250, ps, st; initialdata=test[:, 1])
```

We can now visualize the results

```julia
using Plots
plot(transpose(output); layout=(3, 1), label="predicted");
plot!(transpose(test); layout=(3, 1), label="actual")
```

![lorenz_basic](https://user-images.githubusercontent.com/10376688/166227371-8bffa318-5c49-401f-9c64-9c71980cb3f7.png)

One can also visualize the phase space of the attractor and the
comparison with the actual one:

```julia
plot(transpose(output)[:, 1],
    transpose(output)[:, 2],
    transpose(output)[:, 3];
    label="predicted")
plot!(transpose(test)[:, 1], transpose(test)[:, 2], transpose(test)[:, 3]; label="actual")
```

![lorenz_attractor](https://user-images.githubusercontent.com/10376688/81470281-5a34b580-91ea-11ea-9eea-d2b266da19f4.png)

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
and [RSC4Earth](https://rsc4earth.de/) for supporting the current progress
on the library.
