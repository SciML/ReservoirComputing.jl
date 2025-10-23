# Reservoir Computing using Cellular Automata

We showcase how to use reservoir computing models with cellular automata (ReCA)
with ReservoirComputing.jl. While introduced in [Yilmaz2014](@cite) [Margem2017](@cite),
the implementation in this package follows [Nichele2017](@cite). To showcase ReCA models
we show how to solve the 5 bit memory task.

## 5 bit memory task

We read the data can be read as follows:

```@example reca
using DelimitedFiles

input = readdlm("./5bitinput.txt", ',', Float64)
output = readdlm("./5bitoutput.txt", ',', Float64)
```

To use a ReCA model, it is necessary to define the rule one intends to use.
To do so, ReservoirComputing.jl leverages
[CellularAutomata.jl](https://github.com/MartinuzziFrancesco/CellularAutomata.jl)
that needs to be called as well to define the `RECA` struct:

```@example reca
using ReservoirComputing, CellularAutomata, Random
Random.seed!(42)
rng = MersenneTwister(17)

ca = DCA(90)
```

To define the ReCA model, it suffices to call:

```@example reca
reca = RECA(4, 4, DCA(90);
           generations=16,
           input_encoding=RandomMapping(16, 40))
ps, st = setup(rng, reca)
```
After this, the training can be performed with the chosen method.

```@example reca
ps, st = train!(reca, input, output, ps, st, StandardRidge(0.00001))
```

We are going to test the recall ability of the model, feeding the input data
and investigating whether the predicted output equals the output data.

```@example reca
_, st0 = setup(rng, reca) #reset the first ca state
pred_out, st = predict(reca, input, ps, st0)
final_pred = convert(AbstractArray{Float32}, pred_out .> 0.5)

final_pred == output
```
