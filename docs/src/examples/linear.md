# Using different linear methods 

The standard implementation of ```ESNtrain``` uses Ridge Regression as the method of choice for the training of the ESN but there are other linear methods available. Leveraging [MLJLinearModels](https://alan-turing-institute.github.io/MLJLinearModels.jl/stable/) ReservoirComputing gives the possibility to train ESNs using a vast range of linear models. Using the same task as before, predictiong the Lorenz system, we will illustrate how to use ```ESNtrain``` with Lasso, Elastic Net and regression with Huber loss function.

First we obtain the data in the usual way:

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

And we can also use the same parameters 

```julia
using ReservoirComputing
#model parameters
approx_res_size = 300 
radius = 1.2 
activation = tanh 
degree = 6 
sigma = 0.1 
alpha = 1.0 
nla_type = NLAT2() 
extended_states = false 

#create echo state network  
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

To obtain ```W_out``` we can define a ```linear_model``` struct using one of the implemented constructors:
- ```Ridge()```
- ```Lasso()```
- ```ElastNet()```
- ```RobustHuber()```

Each of them takes as input the regularization coefficient(s) and a MLJLinearModels.Solver as solver. Let's see a couple of examples. Using Ridge():

```julia
using MLJLinearModels
linear_model = Ridge(beta, Analytical())
W_out = ESNtrain(linear_model, esn)
output = ESNpredict(esn, predict_len, W_out)
```
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![esnRidgefixedseed](https://user-images.githubusercontent.com/10376688/90960134-d30d0400-e49f-11ea-847b-6bed1b04c201.png)

Using Lasso():

```julia
beta = 0.0001
linear_model = Lasso(beta, ProxGrad(max_iter = 10000))
W_out = ESNtrain(linear_model, esn)
output = ESNpredict(esn, predict_len, W_out)
```
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![esnLassofixedseed](https://user-images.githubusercontent.com/10376688/90960194-4747a780-e4a0-11ea-9e81-37f12624c503.png)

Since in the linear model struct we used a MLJLinearModels.Solver we can specify any parameter necessary, like in this case the ```max_iter```.

Using ElastNet():

```julia
lambda = 0.1
gamma = 0.0001
linear_model = ElastNet(lambda, gamma, ProxGrad(max_iter = 10000))
W_out = ESNtrain(linear_model, esn)
output = ESNpredict(esn, predict_len, W_out)
```
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![esnElastNetfixedseed](https://user-images.githubusercontent.com/10376688/90960243-98579b80-e4a0-11ea-86ae-2cc9c666d1d9.png)

Using RobustHuber():

```julia
delta = 0.5
lambda = 0.1
gamma = 0.0
linear_model = RobustHuber(lambda, gamma, LBFGS())
W_out = ESNtrain(linear_model, esn)
output = ESNpredict(esn, predict_len, W_out)
```
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```

![esnHuberfixedseed](https://user-images.githubusercontent.com/10376688/90960413-9cd08400-e4a1-11ea-83c3-b3c6dbe78cd5.png)

For the complete list of solvers please refer to [MLJLinearModels solvers](https://alan-turing-institute.github.io/MLJLinearModels.jl/stable/solvers/)
