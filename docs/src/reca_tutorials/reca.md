# Reservoir Computing using Cellular Automata

Reservoir Computing based on Elementary Cellular Automata (ECA) has been recently introduced. Dubbed as ReCA [^1][^2] it proposed the advantage of storing the reservoir states as binary data. Less parameter tuning represents another advantage of this model. The architecture implemented in ReservoirComputing.jl follows [^3] which build over the original implementation, improving the results. It is strongly suggested to go through the paper to get a solid understanding of the model before delving into experimentation with the code.

To showcase how to use this models this page illustrates the performance of ReCA in the 5 bit memory task [^4]. The script for the example and companion data can be found [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/tree/main/reca).

## 5 bit memory task
The data can be read as follows:
```@example reca
using DelimitedFiles

input = readdlm("./5bitinput.txt", ',', Float32)
output = readdlm("./5bitoutput.txt", ',', Float32)
```

To use a ReCA model it is necessary to define the rule one intends to use. To do so ReservoirComputing.jl leverages [CellularAutomata.jl](https://github.com/MartinuzziFrancesco/CellularAutomata.jl) that needs to be called as well to define the `RECA` struct:
```@example reca
using ReservoirComputing, CellularAutomata

ca = DCA(90)
```

To define the ReCA model it suffices to call:
```@example reca
reca = RECA(input, ca; 
    generations = 16,
    input_encoding = RandomMapping(16, 40))
```

After the training can be performed with the chosen method. 
```@example reca
output_layer = train(reca, output, StandardRidge(0.00001))
```

The prediction in this case will be a `Predictive()` with the input data equal to the training data. In addition, to test the 5 bit memory task, a conversion from Float to Bool is necessary (at the moment we are aware of a bug that doesn't allow to input boolean data to the RECA models):
```@example reca
prediction = reca(Predictive(input), output_layer)
final_pred = convert(AbstractArray{Float32}, prediction .> 0.5)

final_pred == output
```

[^1]: Yilmaz, Ozgur. "Reservoir computing using cellular automata." arXiv preprint arXiv:1410.0162 (2014).

[^2]: Margem, Mrwan, and Ozgür Yilmaz. "An experimental study on cellular automata reservoir in pathological sequence learning tasks." (2017).

[^3]: Nichele, Stefano, and Andreas Molund. "Deep reservoir computing using cellular automata." arXiv preprint arXiv:1703.02806 (2017).

[^4]: Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.
