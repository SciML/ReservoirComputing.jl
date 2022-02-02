# Using Different Reservoir Drivers
While the original implementation of the Echo State Network implemented the model using the equations of Recurrent Neural Networks to obtain non linearity in the reservoir, other variations have been proposed in recent years. More specifically the different drivers implemented in ReservoirComputing.jl are the multiple activation function RNN `MRNN()` and the Gated Recurrent Unit `GRU()`. To change them it suffice to give the chosen method to the `ESN` keyword argument `reservoir_driver`. In this section some example of their usage will be given, as well as a quick introduction to their euqations.

## Multiple Activation Function RNN
Based on the double activation function ESN (DAFESN) proposed in [^1], the Multiple Activation Function ESN expands the idea and allows a custom number of activation functions to be used in the reservoir dynamics. This can be thought as a linear combination of multiple activation functions with corresponding parameters.
```math
\mathbf{x}(t+1) = (1-\alpha)\mathbf{x}(t) + \lambda_1 f_1(\mathbf{W}\mathbf{x}(t)+\mathbf{W}_{in}\mathbf{u}(t)) + \dots + \lambda_D f_D(\mathbf{W}\mathbf{x}(t)+\mathbf{W}_{in}\mathbf{u}(t))
```
where ``D`` is the number of activation function and respective parameters chosen.

The method to call to use the mutliple activation function ESN is `MRNN(activation_function, leaky_coefficient, scaling_factor)`. The arguments can be used as both `args` or `kwargs`. `activation_function` and `scaling_factor` have to be vectors (or tuples) containing the chosen activation functions and respective scaling factors (``f_1,...,f_D`` and ``\lambda_1,...,\lambda_D`` following the nomenclature introduced above). The leaky_coefficient represents ``\alpha`` and it is a single value. 

Starting the example, the data used is based on the following function based on the DAFESN paper [^1]. A full script of the example is available [here](scripts/change_drivers_mrnn.jl).
```julia
u(t) = sin(t)+sin(0.51*t)+sin(0.22*t)+sin(0.1002*t)+sin(0.05343*t)
```

For this example the type of prediction will be one step ahead. The metric used to assure a good prediction is going to be the normalized root-mean-square deviation `rmsd` from [StatsBase](https://juliastats.org/StatsBase.jl/stable/). Like in the other examples first it is needed to gather the data:
```julia
data = u.(collect(0.0:0.01:500))
training_input = reduce(hcat, data[shift:shift+train_len-1])
training_target = reduce(hcat, data[shift+1:shift+train_len])
testing_input = reduce(hcat, data[shift+train_len:shift+train_len+predict_len-1])
testing_target = reduce(hcat, data[shift+train_len+1:shift+train_len+predict_len])
```

In order to follow the paper more closely it is necessary to define a couple of activation functions. The numbering of them follows the ones in the paper. Of course one can also use any function, custom defined, available in the base language or any activation function from [NNlib](https://fluxml.ai/Flux.jl/stable/models/nnlib/#Activation-Functions).
```julia
f2(x) = (1-exp(-x))/(2*(1+exp(-x)))
f3(x) = (2/pi)*atan((pi/2)*x)
f4(x) = x/sqrt(1+x*x)
```

It is now possible to build different drivers, using the paramters suggested by the paper. Also in this instance the numbering follows the test cases of the paper. In the end a simple for loop is implemented to compare the different drivers and activation functions.
```julia
using Reservoir Computing, Random

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
    esn = ESN(100, training_input,
        input_init = WeightedLayer(scaling=0.3),
        reservoir_init = RandSparseReservoir(radius=0.4),
        reservoir_driver = case,
        states_type = ExtendedStates())
    wout = train(esn, training_target, StandardRidge(10e-6))
    output = esn(Predictive(testing_input), wout)
    println(rmsd(testing_target, output, normalize=true))
end
```
```
1.2859434466604239e-5
2.1753694726497823e-5
2.9563481223700186e-5
2.5164499914117052e-5
```

In this example it is also possible to observe the input of parameters to the methods `RNN()` `MRNN()` both by argument and by keyword argument.

## Gated Recurrent Unit
Gated Recurrent Units (GRUs) [^2] have been proposed in more recent years with the intent of limiting notable problems of RNNs, like the vanishing gradient. This change in the underlying equations can be easily transported in the Reservoir Computing paradigm, switching the RNN equations in the reservoir with the GRU equations. This approach has been explored in [^3] and [^4]. Different variations of GRU have been proposed [^5][^6]; this section is subdivided into different sections that go in detail about the governing equations and the implementation of them into ReservoirComputing.jl. Like before, to access the GRU reservoir driver it suffice to change the `reservoir_diver` keyword argument for `ESN` with `GRU()`. All the variations that are going to be presented can be used in this package by leveraging the keyword argument `variant` in the method `GRU()` and specifying the chosen variant: `Variant1()`, `Variant2()`, `Variant3()` or `Minimal()`. The default is set to the standard version `FullyGated()`. The first section will go in more detail about the default of the `GRU()` method, and the following ones will refer to it to minimize repetitions.

### Standard GRU
The equations for the standard GRU are as follows:
```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r_{\text{in}}\mathbf{u}(t)+\mathbf{W}^r\mathbf{x}(t-1)+\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z_{\text{in}}\mathbf{u}(t)+\mathbf{W}^z\mathbf{x}(t-1)+\mathbf{b}_z) \\
\tilde{\mathbf{x}}(t) = \text{tanh}(\mathbf{W}_{in}\mathbf{u}(t)+\mathbf{W}(\mathbf{r}(t) \odot \mathbf{x}(t-1))+\mathbf{b}) \\
\mathbf{x}(t) = \mathbf{z}(t) \odot \mathbf{x}(t-1)+(1-\mathbf{z}(t)) \odot \tilde{\mathbf{x}}(t)
```

Going over the `GRU` keyword argument it will be explained how to feed the desired input to the model. 
 - `activation_function` is a vector with default values `[NNlib.sigmoid, NNlib.sigmoid, tanh]`. This argument controls the activation functions of the GRU, going from top to bottom. Changing the first element corresponds in changing the activation function for ``\mathbf{r}(t)``.
 - `layer_init` is a vector with default values `fill(DenseLayer(), 5)`. This keyword argument controls the ``\mathbf{W}_{\text{in}}``s and the ``\mathbf{b}``s going from top to bottom, left to right. For example, changing the first element will change ``\mathbf{W}^r_{\text{in}}``, changing the second will change ``\mathbf{b}_r`` and so on.
 - `reservoir_init` is a vector with default value `fill(RandSparseReservoir(), 2)`. In a similar fashion to `layer_init`, this keyword argument controls the reservoir matrix construction in a top to bottom order. 
 - `variant` as already illustrated controls the GRU variant. The default value is set to `FullyGated()`.
 
It is important to notice that `layer_init` and `reservoir_init` control every layer except ``\mathbf{W}_{in}`` and ``\mathbf{W}``. These arguments are given as input to the `ESN()` call as usual.

### Type 1
The first variation of the GRU is dependent only on the previous hidden state and the bias:
```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r\mathbf{x}(t-1)+\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z\mathbf{x}(t-1)+\mathbf{b}_z) \\
```

This means that `layer_init` is 3-dimensional instead of 5 given the absence of ``\mathbf{W}_{in}``s.

### Type 2
The second variation only depends on the previous hiddens state:
```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r\mathbf{x}(t-1)) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z\mathbf{x}(t-1)) \\
```

Here `layer_init` only has one element, and it control the bias vector of ``\tilde{\mathbf{x}}(t)``.
### Type 3
The final variation before the minimal one depends only on the biases
```math
\mathbf{r}(t) = \sigma (\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{b}_z) \\
```

This means that `layer_init` is 3-dimensional and it controls only the bias vectors. 
### Minimal 
The minimal GRU variation merges two gates into one:
```math
\mathbf{f}(t) = \sigma (\mathbf{W}^f_{\text{in}}\mathbf{u}(t)+\mathbf{W}^f\mathbf{x}(t-1)+\mathbf{b}_f) \\
\tilde{\mathbf{x}}(t) = \text{tanh}(\mathbf{W}_{in}\mathbf{u}(t)+\mathbf{W}(\mathbf{f}(t) \odot \mathbf{x}(t-1))+\mathbf{b}) \\
\mathbf{x}(t) = (1-\mathbf{f}(t)) \odot \mathbf{x}(t-1) + \mathbf{f}(t) \odot \tilde{\mathbf{x}}(t)
```

As a consequence `layer_init` is 3-dimensional and `reservoir_init` is 1-dimensional


### Examples




[^1]: Lun, Shu-Xian, et al. "_A novel model of leaky integrator echo state network for time-series prediction._" Neurocomputing 159 (2015): 58-66.
[^2]: Cho, Kyunghyun, et al. “_Learning phrase representations using RNN encoder-decoder for statistical machine translation._” arXiv preprint arXiv:1406.1078 (2014).
[^3]: Wang, Xinjie, Yaochu Jin, and Kuangrong Hao. "_A Gated Recurrent Unit based Echo State Network._" 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020.
[^4]: Di Sarli, Daniele, Claudio Gallicchio, and Alessio Micheli. "_Gated Echo State Networks: a preliminary study._" 2020 International Conference on INnovations in Intelligent SysTems and Applications (INISTA). IEEE, 2020.
[^5]: Dey, Rahul, and Fathi M. Salem. "_Gate-variants of gated recurrent unit (GRU) neural networks._" 2017 IEEE 60th international midwest symposium on circuits and systems (MWSCAS). IEEE, 2017.
[^6]: Zhou, Guo-Bing, et al. "_Minimal gated unit for recurrent neural networks._" International Journal of Automation and Computing 13.3 (2016): 226-234.
