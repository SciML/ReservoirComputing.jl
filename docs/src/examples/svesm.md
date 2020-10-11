# SVESM

Leveraging the similarities between kernel methods and Reservoir Computing, the paper \[1\] introduced the concept of Support Vector Echo State Machines (SVESMs). Using the package [LIBSVM](https://github.com/JuliaML/LIBSVM.jl) the SVESMs are implemented in ReservoirComputing.jl. We will give an example of usage introducing the direct predict function as well. The goal is again the prediction of the Lorenz system.

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


## References

[1]: Shi, Zhiwei, and Min Han. "Support vector echo-state machine for chaotic time-series prediction." IEEE Transactions on Neural Networks 18.2 (2007): 359-372.
