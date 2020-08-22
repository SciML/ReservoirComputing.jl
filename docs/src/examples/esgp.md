# ESGP 

The linear nature of the ESN training allows for a solution obtained by Gaussian Regression. This was the main idea behind the paper \[1\] that details the implementation of the Echo State Gaussian Processes that are present in ReservoirComputing. Using the same example as before, prediction of the Lorenz system, we are going to show how to use this specific model.

```julia
using ParameterizedFunctions, OrdinaryDiffEq

#lorenz system parameters
u0 = [1.0,0.0,0.0]                       
tspan = (0.0,1000.0)                      
p = [10.0,28.0,8/3]

#define lorenz system 
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

#solve system
prob = ODEProblem(lorenz, u0, tspan, p)  
sol = solve(prob, ABM54(), dt=0.02)   
v = sol.u
data = Matrix(hcat(v...))

shift = 300 
train_len = 5000
predict_len = 1250

#get data
train = data[:, shift:shift+train_len-1]
test = data[:, shift+train_len:shift+train_len+predict_len-1]
```

Now we can define the paramenters and create the ESN in the usual way:

```julia
using ReservoirComputing
#model parameters
approx_res_size = 300 
radius = 1.2 
activation = tanh 
degree = 6
sigma = 0.1 
beta = 0.0 
alpha = 1.0 
nla_type = NLADefault() 
extended_states = true 

#create echo state network  
Random.seed!(42) #fixed seed for reproducibility
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation = activation,
    sigma = sigma,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)
```
Using the package [GaussianProcesses](https://stor-i.github.io/GaussianProcesses.jl/latest/) we were able to implement a training and a predict function for the ESN. In order to use them it is needed to import the package.

```julia
using GaussianProcesses
mean = MeanZero()
kernel = SE(1.0, 1.0)
gp = ESGPtrain(esn, mean, kernel, lognoise = -2.0, optimize = false);
output, sigmas = ESGPpredict(esn, predict_len, gp)
```
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![esgpfixedseed](https://user-images.githubusercontent.com/10376688/90963508-6fdb9b80-e4b8-11ea-98ea-a45980f33cb6.png)

Since the implementation of this model is based on an external package the user is free to choose a different [mean](https://stor-i.github.io/GaussianProcesses.jl/latest/mean/) or [kernel](https://stor-i.github.io/GaussianProcesses.jl/latest/kernels/), as well as using different input layers and reservoirs as defined before.

## References

[1]: Chatzis, Sotirios P., and Yiannis Demiris. "Echo state Gaussian process." IEEE Transactions on Neural Networks 22.9 (2011): 1435-1445.
