# ReservoirComputing.jl
Reservoir computing utilities

## Echo State Network

The code is based on the original [paper](http://www.scholarpedia.org/article/Echo_state_network) by Jaeger, with a few construction changes found in the literature. The reservoir implementation is based on the code used in the following [paper](https://arxiv.org/pdf/1906.08829.pdf), as well as the non linear transformation algorithms T1, T2 and T3, the first of which was introduced [here](https://www.researchgate.net/publication/322457145_Model-Free_Prediction_of_Large_Spatiotemporally_Chaotic_Systems_from_Data_A_Reservoir_Computing_Approach).

The primary goal is to replicate [these](https://arxiv.org/pdf/1710.07313.pdf) results, so the parameters are set as they are described in the paper. The results behave as expected, being able to reproduce the timeseries in the short term and reproducing the climate on the long term. To actually be sure of the reproduction of the results of the paper first I'll have to calculate the Lyapunov exponents thou. Below are the results for the Lorenz System

![Lorenz](https://github.com/MartinuzziFrancesco/EchoStateNetwork/blob/master/comp.png)

It's also interesting to taka a look at the attractors:

![attractors](https://github.com/MartinuzziFrancesco/EchoStateNetwork/blob/master/attractor_com.png)

The predicted one actually shows a similar behaviour to the real one.

After further verifying the correctness of my implementation the goal is to undergo a deep analisys on the construction choices and to implement different systems as reservoir, based on Cellular Automata (ex: The Game of Life), as described in [this](https://arxiv.org/pdf/1410.0162.pdf) paper. 

To do list
* Calculate Lyapunov exponents 
* Implement variable number of outputs as in [this](https://aip.scitation.org/doi/10.1063/1.4979665) paper
* Study the difference in the non linear transformation algorithms
* Implement different systems for the reservoir
