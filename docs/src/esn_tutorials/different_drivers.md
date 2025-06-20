# Using Different Reservoir Drivers

While the original implementation of the Echo State Network implemented the model using the equations of Recurrent Neural Networks to obtain non-linearity in the reservoir, other variations have been proposed in recent years. More specifically, the different drivers implemented in ReservoirComputing.jl are the multiple activation function RNN `MRNN()` and the Gated Recurrent Unit `GRU()`. To change them, it suffices to give the chosen method to the `ESN` keyword argument `reservoir_driver`. In this section, some examples, of their usage will be given, as well as a brief introduction to their equations.

## Multiple Activation Function RNN

Based on the double activation function ESN (DAFESN) proposed in [Lun2015](@cite), the Multiple Activation Function ESN expands the idea and allows a custom number of activation functions to be used in the reservoir dynamics. This can be thought of as a linear combination of multiple activation functions with corresponding parameters.

```math
\mathbf{x}(t+1) = (1-\alpha)\mathbf{x}(t) + \lambda_1 f_1(\mathbf{W}\mathbf{x}(t)+\mathbf{W}_{in}\mathbf{u}(t)) + \dots + \lambda_D f_D(\mathbf{W}\mathbf{x}(t)+\mathbf{W}_{in}\mathbf{u}(t))
```

where ``D`` is the number of activation functions and respective parameters chosen.

The method to call to use the multiple activation function ESN is `MRNN(activation_function, leaky_coefficient, scaling_factor)`. The arguments can be used as both `args` and `kwargs`. `activation_function` and `scaling_factor` have to be vectors (or tuples) containing the chosen activation functions and respective scaling factors (``f_1,...,f_D`` and ``\lambda_1,...,\lambda_D`` following the nomenclature introduced above). The `leaky_coefficient` represents ``\alpha`` and it is a single value.

Starting with the example, the data used is based on the following function based on the DAFESN paper [Lun2015](@cite).

```@example mrnn
u(t) = sin(t) + sin(0.51 * t) + sin(0.22 * t) + sin(0.1002 * t) + sin(0.05343 * t)
```

For this example, the type of prediction will be one step ahead. The metric used to assure a good prediction will be the normalized root-mean-square deviation `rmsd` from [StatsBase](https://juliastats.org/StatsBase.jl/stable/). Like in the other examples, first it is needed to gather the data:

```@example mrnn
train_len = 3000
predict_len = 2000
shift = 1

data = u.(collect(0.0:0.01:500))
training_input = reduce(hcat, data[shift:(shift + train_len - 1)])
training_target = reduce(hcat, data[(shift + 1):(shift + train_len)])
testing_input = reduce(hcat,
    data[(shift + train_len):(shift + train_len + predict_len - 1)])
testing_target = reduce(hcat,
    data[(shift + train_len + 1):(shift + train_len + predict_len)])
```

To follow the paper more closely, it is necessary to define a couple of activation functions. The numbering of them follows the ones in the paper. Of course, one can also use any custom-defined function, available in the base language or any activation function from [NNlib](https://fluxml.ai/NNlib.jl/stable/reference/#Activation-Functions).

```@example mrnn
f2(x) = (1 - exp(-x)) / (2 * (1 + exp(-x)))
f3(x) = (2 / pi) * atan((pi / 2) * x)
f4(x) = x / sqrt(1 + x * x)
```

It is now possible to build different drivers, using the parameters suggested by the paper. Also, in this instance, the numbering follows the test cases of the paper. In the end, a simple for loop is implemented to compare the different drivers and activation functions.

```@example mrnn
using ReservoirComputing, Random, StatsBase

#fix seed for reproducibility
Random.seed!(42)

#baseline case with RNN() driver. Parameter given as args
base_case = RNN(tanh, 0.85)

#MRNN() test cases
#Parameter given as kwargs
case3 = MRNN(; activation_function=[tanh, f2],
    leaky_coefficient=0.85,
    scaling_factor=[0.5, 0.3])

#Parameter given as kwargs
case4 = MRNN(; activation_function=[tanh, f3],
    leaky_coefficient=0.9,
    scaling_factor=[0.45, 0.35])

#Parameter given as args
case5 = MRNN([tanh, f4], 0.9, [0.43, 0.13])

#tests
test_cases = [base_case, case3, case4, case5]
for case in test_cases
    esn = ESN(training_input, 1, 100;
        input_layer=weighted_init(; scaling=0.3),
        reservoir=rand_sparse(; radius=0.4),
        reservoir_driver=case,
        states_type=ExtendedStates())
    wout = train(esn, training_target, StandardRidge(10e-6))
    output = esn(Predictive(testing_input), wout)
    println(rmsd(testing_target, output; normalize=true))
end
```

In this example, it is also possible to observe the input of parameters to the methods `RNN()` `MRNN()`, both by argument and by keyword argument.

## Gated Recurrent Unit

Gated Recurrent Units (GRUs) [Cho2014](@cite) have been proposed in more recent years with the intent of limiting notable problems of RNNs, like the vanishing gradient. This change in the underlying equations can be easily transported into the Reservoir Computing paradigm, by switching the RNN equations in the reservoir with the GRU equations. This approach has been explored in [Wang2020](@cite) and [Sarli2020](@cite). Different variations of GRU have been proposed [Dey2017](@cite); this section is subdivided into different sections that go into detail about the governing equations and the implementation of them into ReservoirComputing.jl. Like before, to access the GRU reservoir driver, it suffices to change the `reservoir_diver` keyword argument for `ESN` with `GRU()`. All the variations that will be presented can be used in this package by leveraging the keyword argument `variant` in the method `GRU()` and specifying the chosen variant: `FullyGated()` or `Minimal()`. Other variations are possible by modifying the inner layers and reservoirs. The default is set to the standard version `FullyGated()`. The first section will go into more detail about the default of the `GRU()` method, and the following ones will refer to it to minimize repetitions.

### Standard GRU

The equations for the standard GRU are as follows:

```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r_{\text{in}}\mathbf{u}(t)+\mathbf{W}^r\mathbf{x}(t-1)+\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z_{\text{in}}\mathbf{u}(t)+\mathbf{W}^z\mathbf{x}(t-1)+\mathbf{b}_z) \\
\tilde{\mathbf{x}}(t) = \text{tanh}(\mathbf{W}_{in}\mathbf{u}(t)+\mathbf{W}(\mathbf{r}(t) \odot \mathbf{x}(t-1))+\mathbf{b}) \\
\mathbf{x}(t) = \mathbf{z}(t) \odot \mathbf{x}(t-1)+(1-\mathbf{z}(t)) \odot \tilde{\mathbf{x}}(t)
```

Going over the `GRU` keyword argument, it will be explained how to feed the desired input to the model.

  - `activation_function` is a vector with default values `[NNlib.sigmoid, NNlib.sigmoid, tanh]`. This argument controls the activation functions of the GRU, going from top to bottom. Changing the first element corresponds to changing the activation function for ``\mathbf{r}(t)`` and so on.
  - `inner_layer` is a vector with default values `fill(DenseLayer(), 2)`. This keyword argument controls the ``\mathbf{W}_{\text{in}}``s going from top to bottom like before.
  - `reservoir` is a vector with default value `fill(RandSparseReservoir(), 2)`. Similarly to `inner_layer`, this keyword argument controls the reservoir matrix construction in a top to bottom order.
  - `bias` is again a vector with default value `fill(DenseLayer(), 2)`. It is meant to control the ``\mathbf{b}``s, going as usual from top to bottom.
  - `variant` controls the GRU variant. The default value is set to `FullyGated()`.

It is important to notice that `inner_layer` and `reservoir` control every layer except ``\mathbf{W}_{in}`` and ``\mathbf{W}`` and ``\mathbf{b}``. These arguments are given as input to the `ESN()` call as `input_layer`, `reservoir` and `bias`.

The following sections are going to illustrate the variations of the GRU architecture and how to obtain them in ReservoirComputing.jl

### Type 1

The first variation of the GRU is dependent only on the previous hidden state and the bias:

```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r\mathbf{x}(t-1)+\mathbf{b}_r) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z\mathbf{x}(t-1)+\mathbf{b}_z) \\
```

To obtain this variation, it will suffice to set `inner_layer = fill(NullLayer(), 2)` and leaving the `variant = FullyGated()`.

### Type 2

The second variation only depends on the previous hidden state:

```math
\mathbf{r}(t) = \sigma (\mathbf{W}^r\mathbf{x}(t-1)) \\
\mathbf{z}(t) = \sigma (\mathbf{W}^z\mathbf{x}(t-1)) \\
```

Similarly to before, to obtain this variation, it is only required to set `inner_layer = fill(NullLayer(), 2)` and `bias = fill(NullLayer(), 2)` while keeping `variant = FullyGated()`.

### Type 3

The final variation, before the minimal one, depends only on the biases

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

To showcase the use of the `GRU()` method, this section will only illustrate the standard `FullyGated()` version. The full script for this example with the data can be found [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/tree/main/change_drivers/gru).

The data used for this example is the Santa Fe laser dataset [Hbner1989](@cite) retrieved from [here](https://web.archive.org/web/20160427182805/http://www-psych.stanford.edu/%7Eandreas/Time-Series/SantaFe.html). The data is split to account for a next step prediction.

```@example gru
using DelimitedFiles

data = reduce(hcat, readdlm("./data/santafe_laser.txt"))

train_len = 5000
predict_len = 2000

training_input = data[:, 1:train_len]
training_target = data[:, 2:(train_len + 1)]
testing_input = data[:, (train_len + 1):(train_len + predict_len)]
testing_target = data[:, (train_len + 2):(train_len + predict_len + 1)]
```

The construction of the ESN proceeds as usual.

```@example gru
using ReservoirComputing, Random

res_size = 300
res_radius = 1.4

Random.seed!(42)
esn = ESN(training_input, 1, res_size;
    reservoir=rand_sparse(; radius=res_radius),
    reservoir_driver=GRU())
```

The default inner reservoir and input layer for the GRU are the same defaults for the `reservoir` and `input_layer` of the ESN. One can use the explicit call if they choose to.

```@example gru
gru = GRU(; reservoir=[rand_sparse,
        rand_sparse],
    inner_layer=[scaled_rand, scaled_rand])
esn = ESN(training_input, 1, res_size;
    reservoir=rand_sparse(; radius=res_radius),
    reservoir_driver=gru)
```

The training and prediction can proceed as usual:

```@example gru
training_method = StandardRidge(0.0)
output_layer = train(esn, training_target, training_method)
output = esn(Predictive(testing_input), output_layer)
```

The results can be plotted using Plots.jl

```@example gru
using Plots

plot([testing_target' output']; label=["actual" "predicted"],
    plot_title="Santa Fe Laser",
    titlefontsize=20,
    legendfontsize=12,
    linewidth=2.5,
    xtickfontsize=12,
    ytickfontsize=12,
    size=(1080, 720))
```

It is interesting to see a comparison of the GRU driven ESN and the standard RNN driven ESN. Using the same parameters defined before it is possible to do the following

```@example gru
using StatsBase

esn_rnn = ESN(training_input, 1, res_size;
    reservoir=rand_sparse(; radius=res_radius),
    reservoir_driver=RNN())

output_layer = train(esn_rnn, training_target, training_method)
output_rnn = esn_rnn(Predictive(testing_input), output_layer)

println(msd(testing_target, output))
println(msd(testing_target, output_rnn))
```

## References

```@bibliography
Pages = ["different_drivers.md"]
Canonical = false
```
