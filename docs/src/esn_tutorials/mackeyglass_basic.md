# Mackey Glass Time Series Forecasting on the GPU

This second introductory example showcases the ability of Echo State Networks (ESNs) to forecast another simple low dimensional complex system, but this time the calculations will be performed on the GPU. More specifically this example uses the [Mackey Glass](http://www.scholarpedia.org/article/Mackey-Glass_equation) chaotic time series. The data and parameters for the following code are taken from dr. Mantas Lukoševičius's [website](https://mantas.info/) and they are going to be compared with the [minimalESN.jl](https://mantas.info/wp/wp-content/uploads/simple_esn/minimalESN.jl) code he provides in it. The full script for this example is available [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/blob/main/mackeyglass_basic/mackeyglass_basic.jl). This example was run on Julia v1.7.2.

## Downloading the Data
Instead of leveraging DifferentialEquations.jl this example downloads directly the data from the website. 
```julia
using Downloads, DelimitedFiles, CUDA

data_path = Downloads.download("https://mantas.info/wp/wp-content/uploads/simple_esn/MackeyGlass_t17.txt", 
    string(pwd(),"/MackeyGlass_t17.txt"))
data = CuArray(reduce(hcat, convert(Matrix{Float32}, readdlm(data_path, ','))))
```

Once the data has been obtained it is time to split it into the needed datasets. In the Lorenz example the initial transient of the data was excluded using a `shift` parameter. In this case the ESN itself is going to be washing out the initial data, so a different parameter `washout` is needed. This needs to be taken into consideration when splitting the data, in order to adjust the target data accordingly.
```julia
washout      = 100
train_len    = 2000
predict_len  = 2000

input_data   = data[:, 1:train_len]
target_data  = data[:, washout+2:train_len+1]
test_data    = data[:, train_len+2:train_len+predict_len+1]
```

## Building the Echo State Network
Now that the data has been split into the needed datasets it is necessary to create the ESN. The input layer and matrix construction is a copy-paste from Mantas's code. This section also showcases how it is possible to feed custom built matrices into the `ESN` constructor, without the need to rely exclusively on the reservoirs and layers included in the library.

In addition this example makes use of the `PaddedExtendedStates()` states type. It is also important to point out that the `washout` parameter need to be passed at this stage. The size of the input layer is the size of the input data, one in this case, plus one, to adjust the dimensions for the padding. Since the padding is equal to unity in this example is possible to not pass any value into the states constructor, since the default is 1.0.
```julia
using ReservoirComputing, Random, LinearAlgebra

Random.seed!(42)
res_size = 1000

esn = ESN(input_data; 
    variation = Default(),
    reservoir = RandSparseReservoir(res_size, 1.25, 1.0),
    input_layer = WeightedLayer(1.0),
    reservoir_driver = RNN(leaky_coefficient=0.3),
    nla_type = NLADefault(),
    states_type = PaddedExtendedStates(),
    washout=washout)
```

## Training and Prediction
The training and prediction are similar to the previous example of the Lorenz system. The ESN is going to be trained in one shot using ridge regression. The prediction type is generative, so the predicted value from the model will be fed back into itself. 
```julia
training_method = StandardRidge(1e-8) 
output_layer    = train(esn, target_data, training_method)    
output          = esn(Generative(predict_len), output_layer)
```

In order to assess if the prediction is comparable to the minimalESN code it suffice to run:
```julia
println(sum(abs2.(test_data[:,1:500] .- output[:,1:500]))/500)
```
```
9.488587078654942e-7
```

This result is comparable to the value from the minimalESN.jl that is 1.6798044273046558e-6. It could also be helpful to visualize the results:
```julia
using Plots

plot([test_data' output'], label = ["actual" "predicted"], 
    plot_title="Makey Glass Time Series",
    titlefontsize=20,
    legendfontsize=12,
    linewidth=2.5,
    xtickfontsize = 12,
    ytickfontsize = 12,
    size=(1080, 720))
```
![mackeyglass_basic](images/mackeyglass_basic.png)
