# Deep Echo State Networks

Deep Echo State Network architectures started to gain some traction recently. In this guide we illustrate how it is possible to use ReservoirComputing.jl to build a deep ESN. 

The network implemented in this library is taken from [^1]. It works by stacking reservoirs on top of each other, feeding the output on one in the next. The states are obtained by merging all the inner states of the stacked reservoirs. For a more in depth explanation refer to the paper linked above. The full script for this example can be found [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/blob/main/deep-esn/deepesn.jl). This example was run on Julia v1.7.2.

## Lorenz Example
For this example we are going to reuse the Lorenz data used in the [Lorenz System Forecasting](@ref) example.
```@example deep_lorenz
using OrdinaryDiffEq

#define lorenz system
function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

#solve and take data
prob = ODEProblem(lorenz!, [1.0,0.0,0.0], (0.0,200.0))
data = solve(prob, ABM54(), dt=0.02)

#determine shift length, training length and prediction length
shift = 300
train_len = 5000
predict_len = 1250

#split the data accordingly
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]
test_data = data[:,shift+train_len+1:shift+train_len+predict_len]
```

Again, it is *important* to notice that the data needs to be formatted in a matrix with the features as rows and time steps as columns like it is done in this example. This is needed even if the time series consists of single values. 

The construction of the ESN is also really similar. The only difference is that the reservoir can be fed as an array of reservoirs. 
```@example deep_lorenz
using ReservoirComputing

reservoirs = [RandSparseReservoir(99, radius=1.1, sparsity=0.1),
    RandSparseReservoir(100, radius=1.2, sparsity=0.1),
    RandSparseReservoir(200, radius=1.4, sparsity=0.1)]

esn = ESN(input_data; 
    variation = Default(),
    reservoir = reservoirs,
    input_layer = DenseLayer(),
    reservoir_driver = RNN(),
    nla_type = NLADefault(),
    states_type = StandardStates())
```

As it is possible to see, different sizes can be chosen for the different reservoirs. The input layer and bias can also be given as vectors, but of course they have to be of the same size of the reservoirs vector. If they are not passed as a vector, the value passed is going to be used for all the layers in the deep ESN.

In addition to using the provided functions for the construction of the layers the user can also choose to build their own matrix, or array of matrices, and feed that into the `ESN` in the same way.

The training and prediction follows the usual framework:
```@example deep_lorenz
training_method = StandardRidge(0.0) 
output_layer = train(esn, target_data, training_method)

output = esn(Generative(predict_len), output_layer)
```
Plotting the results:
```@example deep_lorenz
using Plots

ts = 0.0:0.02:200.0
lorenz_maxlyap = 0.9056
predict_ts = ts[shift+train_len+1:shift+train_len+predict_len]
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

Note that there is a known bug at the moment with using `WeightedLayer` as the input layer with the deep ESN. We are in the process of investigating and solving it. The leak coefficient for the reservoirs has to always be the same with the current implementation. This is also something we are actively looking into expanding.

## Documentation
[^1]: Gallicchio, Claudio, and Alessio Micheli. "_Deep echo state network (deepesn): A brief survey._" arXiv preprint arXiv:1712.04323 (2017).
