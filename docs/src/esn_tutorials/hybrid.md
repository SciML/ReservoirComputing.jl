# Hybrid Echo State Networks
Following the idea of giving physical information to machine learning models the hybrid echo state networks [^1] try to achieve this results by feeding model data into the ESN. In this example it is explained how to create and leverage such models in ReservoirComputing.jl. The full script for this example is available [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/blob/main/hybrid/hybrid.jl). This example was run on Julia v1.7.2.

## Generating the data
For this example we are going to forecast the Lorenz system. As usual the data is generated leveraging `DifferentialEquations.jl`:
```@example hybrid
using DifferentialEquations

u0 = [1.0,0.0,0.0]                       
tspan = (0.0,1000.0)  
datasize = 100000
tsteps = range(tspan[1], tspan[2], length = datasize)                   

function lorenz(du,u,p,t)
    p = [10.0,28.0,8/3]
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

ode_prob = ODEProblem(lorenz, u0, tspan)
ode_sol = solve(ode_prob, saveat = tsteps)
ode_data =Array(ode_sol)

train_len = 10000

input_data  = ode_data[:, 1:train_len]
target_data = ode_data[:, 2:train_len+1]
test_data   = ode_data[:, train_len+1:end][:, 1:1000]

predict_len = size(test_data, 2)
tspan_train = (tspan[1], ode_sol.t[train_len])
```

## Building the Hybrid Echo State Network
In order to feed the data to the ESN it is necessary to create a suitable function.
```@example hybrid
function prior_model_data_generator(u0, tspan, tsteps, model = lorenz)
    prob = ODEProblem(lorenz, u0, tspan) 
    sol = Array(solve(prob, saveat = tsteps))
    return sol
end
```

Given initial condition, time span and time steps this function returns the data for the chosen model. Now, using the `Hybrid` method it is possible to input all this information to the model
```@example hybrid
using ReservoirComputing, Random
Random.seed!(42)

hybrid = Hybrid(prior_model_data_generator, u0, tspan_train, train_len)

esn = ESN(input_data,
    reservoir = RandSparseReservoir(300),
    variation = hybrid)
```

## Training and Prediction
The training and prediction of the Hybrid ESN can proceed as usual:
```@example hybrid
output_layer = train(esn, target_data, StandardRidge(0.3))
output = esn(Generative(predict_len), output_layer)
```

It is now possible to plot the results, leveraging Plots.jl:
```@example hybrid
lorenz_maxlyap = 0.9056
predict_ts = tsteps[train_len+1:train_len+predict_len]
lyap_time = (predict_ts .- predict_ts[1])*(1/lorenz_maxlyap)

p1 = plot(lyap_time, [test_data[1,:] output[1,:]], label = ["actual" "predicted"], 
    ylabel = "x(t)", linewidth=2.5, xticks=false, yticks = -15:15:15);
p2 = plot(lyap_time, [test_data[2,:] output[2,:]], label = ["actual" "predicted"], 
    ylabel = "y(t)", linewidth=2.5, xticks=false, yticks = -20:20:20);
p3 = plot(lyap_time, [test_data[3,:] output[3,:]], label = ["actual" "predicted"], 
    ylabel = "z(t)", linewidth=2.5, xlabel = "max(λ)*t", yticks = 10:15:40);


plot(p1, p2, p3, size=(1080, 720), plot_title = "Lorenz System Coordinates", 
    layout=(3,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20, left_margin=4mm)
```

## Bibliography
[^1]: Pathak, Jaideep, et al. "_Hybrid forecasting of chaotic processes: Using machine learning in conjunction with a knowledge-based model._" Chaos: An Interdisciplinary Journal of Nonlinear Science 28.4 (2018): 041101.
