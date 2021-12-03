# ReservoirComputing.jl
[![Build Status](https://github.com/SciML/ReservoirComputing.jl/workflows/CI/badge.svg)](https://github.com/SciML/ReservoirComputing.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/SciML/ReservoirComputing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ReservoirComputing.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://reservoir.sciml.ai/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://reservoir.sciml.ai/dev/)


![rc_full_logo_large_white_cropped](https://user-images.githubusercontent.com/10376688/144242116-8243f58a-5ac6-4e0e-88d5-3409f00e20b4.png)

ReservoirComputing.jl provides an efficient, modular and easy to use implementation of Reservoir Computing models such as Echo State Networks (ESNs). For information on using this package please refer to the [stable documentation](http://reservoir.sciml.ai/stable/). Use the [in-development documentation](http://reservoir.sciml.ai/dev/) to take a look at at not yet released features.

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
esn = ESN(res_size, input_data; 
          reservoir_init = RandSparseReservoir(radius=1.2, sparsity=6/res_size),
          input_init = WeightedLayer(),
          nla_type = NLAT2())
```

The echo state network can now be trained and tested. If not specified, the training will always be Ordinary Least Squares regression. The full range of training methods is detailed in the documentation.
```julia
output_layer = train(esn, target_data)
output = esn(Generative(predict_len), output_layer)
```

The data is returned as a matrix, `ouput` in the code above, that contains the predicted trajectories. The results can now be easily plotted:
```julia
using Plots
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![lorenz_coord](https://user-images.githubusercontent.com/10376688/81470264-42f5c800-91ea-11ea-98a2-a8a8d7d96155.png)

One can also visualize the phase space of the attractor and the comparison with the actual one:
```julia
plot(transpose(output)[:,1], transpose(output)[:,2], transpose(output)[:,3], label="predicted")
plot!(transpose(test)[:,1], transpose(test)[:,2], transpose(test)[:,3], label="actual")
```
![lorenz_attractor](https://user-images.githubusercontent.com/10376688/81470281-5a34b580-91ea-11ea-9eea-d2b266da19f4.png)
