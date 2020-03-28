[![Build Status](https://travis-ci.com/SciML/ReservoirComputing.jl.svg?branch=master)](https://travis-ci.com/github/SciML/ReservoirComputing.jl)
[![codecov](https://codecov.io/gh/MartinuzziFrancesco/ReservoirComputing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MartinuzziFrancesco/ReservoirComputing.jl)

# ReservoirComputing.jl
Reservoir computing utilities
## Installation
Usual Julia package installation. Run on the Julia terminal:
```julia
julia> using Pkg
julia> Pkg.add("ReservoirComputing")
```
## Echo State Network example

This example and others are contained in the examples folder, that will be updated anytime I find new examples.
To show how to use some of the functions contained in ReservoirComputing.jl we will illustrate an example also shown in literature: reproducing the Lorenz attractor.
First we have to define the Lorenz system and the parameters we are going to use
```julia
using ParameterizedFunctions
using OrdinaryDiffEq
     
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
sol = solve(prob, AB4(), dt=0.02)   
v = sol.u
data = Matrix(hcat(v...))
shift = 1
train_len = 5000
predict_len = 1250
train = data[:, shift:shift+train_len-1]
test = data[:, train_len:train_len+predict_len-1]
```
Now that we have the data we can initialize the parameters for the echo state network
```julia
approx_res_size = 300
radius = 1.2
degree = 6
activation = tanh
sigma = 0.1
beta = 0.0
alpha = 1.0
nonlin_alg = NonLinAlgT2
```
Now it's time to initiate the echo state network
```julia
using ReservoirComputing
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation, #default = tanh
    sigma, #default = 0.1
    alpha, #default = 1.0
    beta, #default = 0.0
    nonlin_alg #default = "None"
    )
```
The echo state network can now be trained and tested:
```julia
W_out = ESNtrain(esn)
output = ESNpredict(esn, predict_len, W_out)
```
ouput is the matrix with the predicted trajectories that can be easily plotted 
```julia
using Plots
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![Lorenz](https://user-images.githubusercontent.com/10376688/72996946-dbaf3600-3dfb-11ea-8d5d-3a7356780b5e.png)

One can also visualize the phase space of the attractor and the comparison with the actual one:
```julia
plot(transpose(output)[:,1], transpose(output)[:,2], transpose(output)[:,3], label="predicted")
plot!(transpose(test)[:,1], transpose(test)[:,2], transpose(test)[:,3], label="actual")
```
![attractor](https://user-images.githubusercontent.com/10376688/72997095-1913c380-3dfc-11ea-9702-a9734a375b96.png)

The results are in line with the literature.

The code is partly based on the original [paper](http://www.scholarpedia.org/article/Echo_state_network) by Jaeger, with a few construction changes found in the literature. The reservoir implementation is based on the code used in the following [paper](https://arxiv.org/pdf/1906.08829.pdf), as well as the non linear transformation algorithms T1, T2 and T3, the first of which was introduced [here](https://www.researchgate.net/publication/322457145_Model-Free_Prediction_of_Large_Spatiotemporally_Chaotic_Systems_from_Data_A_Reservoir_Computing_Approach).


## To do list
* Documentation
* Implement variable number of outputs as in [this](https://aip.scitation.org/doi/10.1063/1.4979665) paper
* Implement different systems for the reservoir (like [this](https://arxiv.org/pdf/1410.0162.pdf) paper)
