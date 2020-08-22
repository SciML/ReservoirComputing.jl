# Double Activation Function ESN
One of the different variations of the ESN implemented in ReservoirComputing is the Double Activation Function ESN (DASFESN) \[1\]. The construrs are implemented in the same way as the ESN ones and the training is done with the same ```ESNtrain``` function. The only differences are the number of parameters and the predict function. Let's use the same Lorenz system example as always to see how to use this model.

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
Now we can define the parameters as before: this time we will have two activation functions instead of one and a linear coefficient for each function defined as ```first_lambda``` and ```second_lambda```. For the second activation function we will use the sigmoid function, that can be either user defined or borrowed from another package. We will use the implementation of NNlib.

```julia
using ReservoirComputing, NNlib
#model parameters
approx_res_size = 300 
radius = 1.2 
degree = 6 
sigma = 0.1 
beta = 0.0 
alpha = 1.0 
nla_type = NLAT2() 
extended_states = false 

first_lambda = 0.8
second_lambda = 0.4
first_activation = tanh
second_activation = σ

#create echo state network  
Random.seed!(42) #fixed seed for reproducibility
esn = dafESN(approx_res_size,
    train,
    degree,
    radius,
    first_lambda,
    second_lambda,
    first_activation = first_activation,
    second_activation = second_activation,
    sigma = sigma,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)
```
```julia
plot(transpose(output),layout=(3,1), label="predicted")
plot!(transpose(test),layout=(3,1), label="actual")
```
![dafesnfixedseed](https://user-images.githubusercontent.com/10376688/90960708-e6ba6980-e4a3-11ea-8390-e29c74931991.png)

In this example we used the standard constructor but the user is free to define the input layer and reservoir in the same way we showed before. Of course one could also train the DAFESN using one of the already illustrated linear models.

## References

[1]: Lun, Shu-Xian, et al. "A novel model of leaky integrator echo state network for time-series prediction." Neurocomputing 159 (2015): 58-66.
