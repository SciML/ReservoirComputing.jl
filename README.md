[![Build Status](https://github.com/SciML/ReservoirComputing.jl/workflows/CI/badge.svg)](https://github.com/SciML/ReservoirComputing.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/SciML/ReservoirComputing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ReservoirComputing.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://reservoir.sciml.ai/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://reservoir.sciml.ai/dev/)

# ReservoirComputing.jl
Reservoir computing utilities
## Installation
Usual Julia package installation. Run on the Julia terminal:
```julia
julia> using Pkg
julia> Pkg.add("ReservoirComputing")
```
## Echo State Network example

This example and others are contained in the examples folder, which will be updated whenever I find new examples.
To show how to use some of the functions contained in ReservoirComputing.jl, we will illustrate it by means of an example also shown in the literature: reproducing the Lorenz attractor.
First, we have to define the Lorenz system and the parameters we are going to use:
```julia
using ParameterizedFunctions
using OrdinaryDiffEq
using ReservoirComputing

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
sol = solve(prob, ABM54(), dt=0.02)   
v = sol.u
data = Matrix(hcat(v...))
shift = 300
train_len = 5000
predict_len = 1250
train = data[:, shift:shift+train_len-1]
test = data[:, shift+train_len:shift+train_len+predict_len-1]
```
Now that we have the datam we can initialize the parameters for the echo state network:
```julia
approx_res_size = 300
radius = 1.2
degree = 6
activation = tanh
sigma = 0.1
beta = 0.0
alpha = 1.0
nla_type = NLAT2()
extended_states = false
```
Now it's time to initiate the echo state network:
```julia
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation = activation, #default = tanh
    alpha = alpha, #default = 1.0
    sigma = sigma, #default = 0.1
    nla_type = nla_type, #default = NLADefault()
    extended_states = extended_states #default = false
    )
```
The echo state network can now be trained and tested:
```julia
W_out = ESNtrain(esn, beta)
output = ESNpredict(esn, predict_len, W_out)
```
ouput is the matrix with the predicted trajectories that can be easily plotted
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

The results are in line with the literature.

The code is partly based on the original [paper](http://www.scholarpedia.org/article/Echo_state_network) by Jaeger, with a few construction changes found in the literature. The reservoir implementation is based on the code used in the following [paper](https://arxiv.org/pdf/1906.08829.pdf), as well as the non-linear transformation algorithms T1, T2, and T3, the first of which was introduced [here](https://www.researchgate.net/publication/322457145_Model-Free_Prediction_of_Large_Spatiotemporally_Chaotic_Systems_from_Data_A_Reservoir_Computing_Approach).

## Citing

If you use this library in your work, please cite:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.05117,
  doi = {10.48550/ARXIV.2204.05117},
  url = {https://arxiv.org/abs/2204.05117},
  author = {Martinuzzi, Francesco and Rackauckas, Chris and Abdelrehim, Anas and Mahecha, Miguel D. and Mora, Karin},
  keywords = {Computational Engineering, Finance, and Science (cs.CE), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ReservoirComputing.jl: An Efficient and Modular Library for Reservoir Computing Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```