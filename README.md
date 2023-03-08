# ReservoirComputing.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/ReservoirComputing/stable/)
 [![arXiv](https://img.shields.io/badge/arXiv-2204.05117-00b300.svg)](https://arxiv.org/abs/2204.05117)

[![codecov](https://codecov.io/gh/SciML/ReservoirComputing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ReservoirComputing.jl)
[![Build Status](https://github.com/SciML/ReservoirComputing.jl/workflows/CI/badge.svg)](https://github.com/SciML/ReservoirComputing.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/db8f91b89a10ad79bbd1d9fdb1340e6f6602a1c0ed9496d4d0.svg)](https://buildkite.com/julialang/reservoircomputing-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

![rc_full_logo_large_white_cropped](https://user-images.githubusercontent.com/10376688/144242116-8243f58a-5ac6-4e0e-88d5-3409f00e20b4.png)

ReservoirComputing.jl provides an efficient, modular and easy to use implementation of Reservoir Computing models such as Echo State Networks (ESNs). For information on using this package please refer to the [stable documentation](https://docs.sciml.ai/ReservoirComputing/stable/). Use the [in-development documentation](https://docs.sciml.ai/ReservoirComputing/dev/) to take a look at at not yet released features.

## Quick Example

To illustrate the workflow of this library we will showcase how it is possible to train an ESN to learn the dynamics of the Lorenz system. As a first step we will need to gather the data. For the `Generative` prediction we need the target data to be one step ahead of the training data:

```julia
using ReservoirComputing, OrdinaryDiffEq

#lorenz system parameters
u0 = [1.0,0.0,0.0]                       
tspan = (0.0,200.0)                      
p = [10.0,28.0,8/3]

#define lorenz system
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end
#solve and take data
prob = ODEProblem(lorenz, u0, tspan, p)  
data = solve(prob, ABM54(), dt=0.02)   

shift = 300
train_len = 5000
predict_len = 1250

#one step ahead for generative prediction
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]

test = data[:,shift+train_len:shift+train_len+predict_len-1]
```
Now that we have the data we can initialize the ESN with the chosen parameters. Given that this is a quick example we are going to change the least amount of possible parameters. For more detailed examples and explanations of the functions please refer to the documentation.
```julia
res_size = 300
esn = ESN(input_data; 
          reservoir = RandSparseReservoir(res_size, radius=1.2, sparsity=6/res_size),
          input_layer = WeightedLayer(),
          nla_type = NLAT2())
```

The echo state network can now be trained and tested. If not specified, the training will always be Ordinary Least Squares regression. The full range of training methods is detailed in the documentation.
```julia
output_layer = train(esn, target_data)
output = esn(Generative(predict_len), output_layer)
```

The data is returned as a matrix, `ouput` in the code above, that contains the predicted trajectories. The results can now be easily plotted (for the actual script used to obtain this plot plese refer to the documentation):
```julia
using Plots
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![lorenz_basic](https://user-images.githubusercontent.com/10376688/166227371-8bffa318-5c49-401f-9c64-9c71980cb3f7.png)

One can also visualize the phase space of the attractor and the comparison with the actual one:
```julia
plot(transpose(output)[:,1], transpose(output)[:,2], transpose(output)[:,3], label="predicted")
plot!(transpose(test)[:,1], transpose(test)[:,2], transpose(test)[:,3], label="actual")
```
![lorenz_attractor](https://user-images.githubusercontent.com/10376688/81470281-5a34b580-91ea-11ea-9eea-d2b266da19f4.png)

## Citing

If you use this library in your work, please cite:

```bibtex
@article{JMLR:v23:22-0611,
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
