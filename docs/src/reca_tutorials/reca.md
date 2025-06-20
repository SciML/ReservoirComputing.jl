# Reservoir Computing using Cellular Automata

Reservoir Computing based on Elementary Cellular Automata (ECA) has been recently introduced. Dubbed as ReCA [Yilmaz2014](@cite) [Margem2017](@cite) it proposed the advantage of storing the reservoir states as binary data. Less parameter tuning represents another advantage of this model. The architecture implemented in ReservoirComputing.jl follows [Nichele2017](@cite) which builds on top of the original implementation, improving the results. It is strongly suggested to go through the paper to get a solid understanding of the model before delving into experimentation with the code.

To showcase how to use these models, this page illustrates the performance of ReCA in the 5 bit memory task.

## 5 bit memory task

The data can be read as follows:

```@example reca
using DelimitedFiles

input = readdlm("./5bitinput.txt", ',', Float64)
output = readdlm("./5bitoutput.txt", ',', Float64)
```

To use a ReCA model, it is necessary to define the rule one intends to use. To do so, ReservoirComputing.jl leverages [CellularAutomata.jl](https://github.com/MartinuzziFrancesco/CellularAutomata.jl) that needs to be called as well to define the `RECA` struct:

```@example reca
using ReservoirComputing, CellularAutomata

ca = DCA(90)
```

To define the ReCA model, it suffices to call:

```@example reca
reca = RECA(input, ca;
    generations=16,
    input_encoding=RandomMapping(16, 40))
```

After this, the training can be performed with the chosen method.

```@example reca
output_layer = train(reca, output, StandardRidge(0.00001))
```

The prediction in this case will be a `Predictive()` with the input data equal to the training data. In addition, to test the 5 bit memory task, a conversion from Float to Bool is necessary (at the moment, we are aware of a bug that doesn't allow boolean input data to the RECA models):

```@example reca
prediction = reca(Predictive(input), output_layer)
final_pred = convert(AbstractArray{Float32}, prediction .> 0.5)

final_pred == output
```

