# Using Different Reservoir Drivers
While the original implementation of the Echo State Network implemented the model using the equations of Recurrent Neural Networks to obtain non linearity in the reservoir, other variations have been proposed in recent years. More specifically the different drivers implemented in ReservoirComputing.jl are the multiple activation function RNN `MRNN()` and the Gated Recurrent Unit `GRU()`. To change them it suffice to give the chosen method to the `ESN` keyword argument `reservoir_driver`. In this section some example of their usage will be given, as well as a quick introduction to their equations.

## Multiple Activation Function RNN
Based on the double activation function ESN (DAFESN) proposed in [^1], the Multiple Activation Function ESN expands the idea and allows a custom number of activation functions to be used in the reservoir dynamics. This can be thought as a linear combination of multiple activation functions with corresponding parameters.
```math
\mathbf{x}(t+1) = (1-\alpha)\mathbf{x}(t) + \lambda_1 f_1(\mathbf{W}\mathbf{x}(t)+\mathbf{W}_{in}\mathbf{u}(t)) + \dots + \lambda_D f_D(\mathbf{W}\mathbf{x}(t)+\mathbf{W}_{in}\mathbf{u}(t))
```
where ``D`` is the number of activation function and respective parameters chosen.

The method to call to use the mutliple activation function ESN is `MRNN(activation_function, leaky_coefficient, scaling_factor)`. The arguments can be used as both `args` or `kwargs`. `activation_function` and `scaling_factor` have to be vectors (or tuples) containing the chosen activation functions and respective scaling factors (``f_1,...,f_D`` and ``\lambda_1,...,\lambda_D`` following the nomenclature introduced above). The leaky_coefficient represents ``\alpha`` and it is a single value. 

Starting the example, the data used is based on the following function based on the DAFESN paper [^1]. A full script of the example is available [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/blob/main/change_drivers/mrnn/mrnn.jl). This example was run on Julia v1.7.2.
```@example mrnn
u(t) = sin(t)+sin(0.51*t)+sin(0.22*t)+sin(0.1002*t)+sin(0.05343*t)
```

For this example the type of prediction will be one step ahead. The metric used to assure a good prediction is going to be the normalized root-mean-square deviation `rmsd` from [StatsBase](https://juliastats.org/StatsBase.jl/stable/). Like in the other examples first it is needed to gather the data:
```@example mrnn
train_len = 3000
predict_len = 2000
shift = 1

data = u.(collect(0.0:0.01:500))
training_input = reduce(hcat, data[shift:shift+train_len-1])
training_target = reduce(hcat, data[shift+1:shift+train_len])
testing_input = reduce(hcat, data[shift+train_len:shift+train_len+predict_len-1])
testing_target = reduce(hcat, data[shift+train_len+1:shift+train_len+predict_len])
```

In order to follow the paper more closely it is necessary to define a couple of activation functions. The numbering of them follows the ones in the paper. Of course one can also use any function, custom defined, available in the base language or any activation function from [NNlib](https://fluxml.ai/Flux.jl/stable/models/nnlib/#Activation-Functions).
```@example mrnn
f2(x) = (1-exp(-x))/(2*(1+exp(-x)))
f3(x) = (2/pi)*atan((pi/2)*x)
f4(x) = x/sqrt(1+x*x)
```

It is now possible to build different drivers, using the parameters suggested by the paper. Also in this instance the numbering follows the test cases of the paper. In the end a simple for loop is implemented to compare the different drivers and activation functions.
```@example mrnn
using ReservoirComputing, Random

#fix seed for reproducibility
Random.seed!(42)

#baseline case with RNN() driver. Parameter given as args
base_case = RNN(tanh, 0.85)

#MRNN() test cases
#Parameter given as kwargs
case3 = MRNN(activation_function=[tanh, f2], 
    leaky_coefficient=0.85, 
    scaling_factor=[0.5, 0.3])

#Parameter given as kwargs
case4 = MRNN(activation_function=[tanh, f3], 
    leaky_coefficient=0.9, 
    scaling_factor=[0.45, 0.35])

#Parameter given as args
case5 = MRNN([tanh, f4], 0.9, [0.43, 0.13])

#tests
test_cases = [base_case, case3, case4, case5]
for case in test_cases
    esn = ESN(training_input,
        input_layer = WeightedLayer(scaling=0.3),
        reservoir = RandSparseReservoir(100, radius=0.4),
        reservoir_driver = case,
        states_type = ExtendedStates())
    wout = train(esn, training_target, StandardRidge(10e-6))
    output = esn(Predictive(testing_input), wout)
    println(rmsd(testing_target, output, normalize=true))
end
```

In this example it is also possible to observe the input of parameters to the methods `RNN()` `MRNN()` both by argument and by keyword argument.

## Gated Recurrent Unit
Gated Recurrent Units (GRUs) [^2] have been proposed in more recent years with the intent of limiting notable problems of RNNs, like the vanishing gradient. This change in the underlying equations can be easily transported in the Reservoir Computing paradigm, switching the RNN equations in the reservoir with the GRU equations. This approach has been explored in [^3] and [^4]. Different variations of GRU have been proposed [^5][^6]; this section is subdivided into different sections that go in detail about the governing equations and the implementation of them into ReservoirComputing.jl. Like before, to access the GRU reservoir driver it suffice to change the `reservoir_diver` keyword argument for `ESN` with `GRU()`. All the variations that are going to be presented can be used in this package by leveraging the keyword argument `variant` in the method `GRU()` and specifying the chosen variant: `FullyGated()` or `Minimal()`. Other variations are possible modifying the inner layers and reservoirs. The default is set to the standard version `FullyGated()`. The first section will go in more detail about the default of the `GRU()` method, and the following ones will refer to it to minimize repetitions. This example was run on Julia v1.7.2.

### Standard GRU
The equations for the standard GRU are as follows:
```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r_{\text{in}}\mathbf{u}(t)+\mathbf{W}^r\mathbf{x}(t-1)+\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z_{\text{in}}\mathbf{u}(t)+\mathbf{W}^z\mathbf{x}(t-1)+\mathbf{b}_z) \\
\tilde{\mathbf{x}}(t) = \text{tanh}(\mathbf{W}_{in}\mathbf{u}(t)+\mathbf{W}(\mathbf{r}(t) \odot \mathbf{x}(t-1))+\mathbf{b}) \\
\mathbf{x}(t) = \mathbf{z}(t) \odot \mathbf{x}(t-1)+(1-\mathbf{z}(t)) \odot \tilde{\mathbf{x}}(t)
```

Going over the `GRU` keyword argument it will be explained how to feed the desired input to the model. 
 - `activation_function` is a vector with default values `[NNlib.sigmoid, NNlib.sigmoid, tanh]`. This argument controls the activation functions of the GRU, going from top to bottom. Changing the first element corresponds in changing the activation function for ``\mathbf{r}(t)`` and so on.
 - `inner_layer` is a vector with default values `fill(DenseLayer(), 2)`. This keyword argument controls the ``\mathbf{W}_{\text{in}}``s going from top to bottom like before.
 - `reservoir` is a vector with default value `fill(RandSparseReservoir(), 2)`. In a similar fashion to `inner_layer`, this keyword argument controls the reservoir matrix construction in a top to bottom order.
 - `bias` is again a vector with default value `fill(DenseLayer(), 2)`. It is meant to control the ``\mathbf{b}``s, going as usual from top to bottom.
 - `variant` as already illustrated controls the GRU variant. The default value is set to `FullyGated()`.
 
It is important to notice that `inner_layer` and `reservoir` control every layer except ``\mathbf{W}_{in}`` and ``\mathbf{W}`` and ``\mathbf{b}``. These arguments are given as input to the `ESN()` call as `input_layer`, `reservoir` and `bias`. 

The following sections are going to illustrate the variations of the GRU architecture and how to obtain them in ReservoirComputing.jl

### Type 1
The first variation of the GRU is dependent only on the previous hidden state and the bias:
```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r\mathbf{x}(t-1)+\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z\mathbf{x}(t-1)+\mathbf{b}_z) \\
```

To obtain this variation it will suffice to set `inner_layer = fill(NullLayer(), 2)` and leaving the `variant = FullyGated()`.

### Type 2
The second variation only depends on the previous hidden state:
```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r\mathbf{x}(t-1)) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z\mathbf{x}(t-1)) \\
```

Similarly to before, to obtain this variation it is only needed to set `inner_layer = fill(NullLayer(), 2)` and `bias = fill(NullLayer(), 2)` while keeping `variant = FullyGated()`.

### Type 3
The final variation before the minimal one depends only on the biases
```math
\mathbf{r}(t) = \sigma (\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{b}_z) \\
```

This means that it is only needed to set `inner_layer = fill(NullLayer(), 2)` and `reservoir = fill(NullReservoir(), 2)` while keeping `variant = FullyGated()`.

### Minimal 
The minimal GRU variation merges two gates into one:
```math
\mathbf{f}(t) = \sigma (\mathbf{W}^f_{\text{in}}\mathbf{u}(t)+\mathbf{W}^f\mathbf{x}(t-1)+\mathbf{b}_f) \\
\tilde{\mathbf{x}}(t) = \text{tanh}(\mathbf{W}_{in}\mathbf{u}(t)+\mathbf{W}(\mathbf{f}(t) \odot \mathbf{x}(t-1))+\mathbf{b}) \\
\mathbf{x}(t) = (1-\mathbf{f}(t)) \odot \mathbf{x}(t-1) + \mathbf{f}(t) \odot \tilde{\mathbf{x}}(t)
```

This variation can be obtained by setting `variation=Minimal()`. The `inner_layer`, `reservoir` and `bias` kwargs this time are **not** vectors, but must be defined like, for example `inner_layer = DenseLayer()` or `reservoir = SparseDenseReservoir()`.

### Examples
To showcase the use of the `GRU()` method this section will only illustrate the standard `FullyGated()` version. The full script for this example with the data can be found [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/blob/main/change_drivers/gru/l). 

The data used for this example is the Santa Fe laser dataset [^7] retrieved from [here](https://web.archive.org/web/20160427182805/http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe.html). The data is split to account for a next step prediction.
```@example gru
using DelimitedFiles

data = reduce(hcat, readdlm("santafe_laser.txt"))

train_len   = 5000
predict_len = 2000

training_input  = data[:, 1:train_len]
training_target = data[:, 2:train_len+1]
testing_input   = data[:,train_len+1:train_len+predict_len]
testing_target  = data[:,train_len+2:train_len+predict_len+1]
```

The construction of the ESN proceeds as usual. 
```@example gru
using ReservoirComputing, Random

res_size = 300
res_radius = 1.4

Random.seed!(42)
esn = ESN(training_input; 
    reservoir = RandSparseReservoir(res_size, radius=res_radius),
    reservoir_driver = GRU())
```

The default inner reservoir and input layer for the GRU are the same defaults for the `reservoir` and `input_layer` of the ESN. One can use the explicit call if they choose so.
```@example gru
gru = GRU(reservoir=[RandSparseReservoir(res_size), 
    RandSparseReservoir(res_size)],
    inner_layer=[DenseLayer(), DenseLayer()])
esn = ESN(training_input; 
    reservoir = RandSparseReservoir(res_size, radius=res_radius),
    reservoir_driver = gru)
```

The training and prediction can proceed as usual:
```@example gru
training_method = StandardRidge(0.0)
output_layer    = train(esn, training_target, training_method)
output          = esn(Predictive(testing_input), output_layer)
```

The results can be plotted using Plots.jl
```@example gru
using Plots

plot([testing_target' output'], label=["actual" "predicted"], 
    plot_title="Santa Fe Laser",
    titlefontsize=20,
    legendfontsize=12,
    linewidth=2.5,
    xtickfontsize = 12,
    ytickfontsize = 12,
    size=(1080, 720))
```

It is interesting to see a comparison of the GRU driven ESN and the standard RNN driven ESN. Using the same parameters defined before it is possible to do the following
```@example gru
using StatsBase

esn_rnn = ESN(training_input; 
    reservoir = RandSparseReservoir(res_size, radius=res_radius),
    reservoir_driver = RNN())

output_layer    = train(esn_rnn, training_target, training_method)
output_rnn      = esn_rnn(Predictive(testing_input), output_layer)

println(msd(testing_target, output))
println(msd(testing_target, output_rnn))
```

[^1]: Lun, Shu-Xian, et al. "_A novel model of leaky integrator echo state network for time-series prediction._" Neurocomputing 159 (2015): 58-66.
[^2]: Cho, Kyunghyun, et al. “_Learning phrase representations using RNN encoder-decoder for statistical machine translation._” arXiv preprint arXiv:1406.1078 (2014).
[^3]: Wang, Xinjie, Yaochu Jin, and Kuangrong Hao. "_A Gated Recurrent Unit based Echo State Network._" 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020.
[^4]: Di Sarli, Daniele, Claudio Gallicchio, and Alessio Micheli. "_Gated Echo State Networks: a preliminary study._" 2020 International Conference on INnovations in Intelligent SysTems and Applications (INISTA). IEEE, 2020.
[^5]: Dey, Rahul, and Fathi M. Salem. "_Gate-variants of gated recurrent unit (GRU) neural networks._" 2017 IEEE 60th international midwest symposium on circuits and systems (MWSCAS). IEEE, 2017.
[^6]: Zhou, Guo-Bing, et al. "_Minimal gated unit for recurrent neural networks._" International Journal of Automation and Computing 13.3 (2016): 226-234.
[^7]: Hübner, Uwe, Nimmi B. Abraham, and Carlos O. Weiss. "_Dimensions and entropies of chaotic intensity pulsations in a single-mode far-infrared NH 3 laser._" Physical Review A 40.11 (1989): 6354.
