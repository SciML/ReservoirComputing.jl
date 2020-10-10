# Using different layers

The composability of the ESN struct allows for the construction of the model with different reservoirs or input layers. In ReservoirComputing.jl there are different implementations of these layers found in the literature, but, of course, one is free to build a custom implementation. Following the prior example, we will continue to use the Lorenz system prediction as our test.

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

## Delay Line Reservoir and dense input layer
In order to change the default reservoir and input layer, we first need to define the ones we want to use. With the same parameters as the example before, we can define the ESN as follows:

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

#define the weight for the reservoir
r= 0.95

#define reservoir and input layer
Random.seed!(42) #fixed seed for reproducibility
W = DLR(approx_res_size, r)
W_in = init_dense_input_layer(approx_res_size, size(train, 1), sigma)

#create echo state network  
esndlr = ESN(W,
    train,
    W_in,
    activation = activation,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)

#training and prediction
W_out = ESNtrain(esndlr, beta)
output = ESNpredict(esndlr, predict_len, W_out)
```
And we can plot the results as before:
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```

![esndlrfixedseed](https://user-images.githubusercontent.com/10376688/90959227-ffbe1d00-e499-11ea-8a0a-c4ba95ddb29a.png)

The reservoir used is taken from the paper \[1\]. The other reservoir illustrated therein are implemented in this package.

## Pseudo SVD reservoir and irrational input layer
Using another architecture just to have more examples, we can define the ESN using the reservoir obtained from the SVD-like method \[2\] and the irrational input layer, as described in \[1\]. We are going to use the same parameters as before, only adding the necessary ones for the construction of the new layers.

```julia
max_value = 1.5
sparsity = 0.15

#constructing the pseudo svd reservoir and irrational input layer
W = pseudoSVD(approx_res_size, max_value, sparsity)
W_in = irrational_sign_input(approx_res_size, size(train, 1), sigma)

esnsvd = ESN(W,
    train,
    W_in,
    activation = activation,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)

#training and prediction
W_out = ESNtrain(esnsvd, beta)
output = ESNpredict(esnsvd, predict_len, W_out)
```
And, of course, the result can be plotted as always:
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![esnsvdfixedseed](https://user-images.githubusercontent.com/10376688/90959522-f59d1e00-e49b-11ea-9b34-9e88ae3b4adf.png)

As we can see, the results can vary wildly from one architecture to another, so be careful with the choice of the layers.

## References

\[1\]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.

\[2\]" Yang, Cuili, et al. "Design of polynomial echo state networks for time series prediction." Neurocomputing 290 (2018): 148-160.
