# ReservoirComputing.jl

ReservoirComputing.jl provides an efficient, modular and easy to use implementation of Reservoir Computing models such as Echo State Networks (ESNs). Reservoir Computing (RC) is an umbrella term used to describe a family of models such as ESNs and Liquid State Machines (LSMs). The key concept is to expand the input data into a higher dimension and use regression in order to train the model; in some ways Reservoir Computers can be considered similar to kernel methods. 

This package is still very much in development, so expect bugs and strange behaviors. If we find something strange don't hesitate to open an issue about it so it can be looked into.

!!! info "Introductory material"
    This library assumes some basic knowledge of Reservoir Computing. For a good introduction, we suggest the following papers: the first two are the seminal papers about ESN and LSM, the others are in-depth review papers that should cover all the needed information. For the majority of the algorithms implemented in this library we cited in the documentation the original work introducing them. If you ever are in doubt about about a method or a function just type ```? function``` in the Julia REPL to read the relevant notes.

    * Jaeger, Herbert: The “echo state” approach to analysing and training recurrent neural networks-with an erratum note.
    * Maass W, Natschläger T, Markram H: Real-time computing without stable states: a new framework for neural computation based on perturbations.
    * Lukoševičius, Mantas: A practical guide to applying echo state networks." Neural networks: Tricks of the trade.
    * Lukoševičius, Mantas, and Herbert Jaeger: Reservoir computing approaches to recurrent neural network training.
    
## Installation
The installation of the package is done following the usual Julia procedure:
```julia
import Pkg; Pkg.add("ReservoirComputing")
```
The support for this library is for Julia v1.6 or greater.

## Features Overview

This library provides multiple ways of training the chosen RC model. More specifically the available algorithms are:
- ```StandardRidge```: a naive implementation of Ridge Regression. The default choice for training.
- ```LinearModel```: a wrap around [MLJLinearModels](https://juliaai.github.io/MLJLinearModels.jl/stable/).
- ```GaussianProcess```: a wrap around [GaussianProcesses](http://stor-i.github.io/GaussianProcesses.jl/latest/).
- ```LIBSVM.AbstractSVR```: a direct call of [LIBSVM](https://github.com/JuliaML/LIBSVM.jl) regression methods.

Also provided are two different ways of doing predictions using RC:
- ```Generative```: the algorithm uses the prediction of the model in the previous step to continue the prediction. It only needs the number of steps as input.
- ```Predictive```: standard Machine Learning type of prediction. Given the features the RC model will return the label/prediction.

It is possible to modify the RC obtained states in the training and prediction step using the following:
- ```StandardStates```: default choice, no changes will be made to the states.
- ```ExtendedStates```: the states are extended using a vertical concatenation with the input data.
- ```PaddedStates```: the states are padded using a vertical concatenation with the chosing padding value
- ```PaddedExtendedStates```: a combination of the first two. First the states are extended and then padded.

In addition another modification is possible through the choice of non linear algorithms:
- ```NLADefault```: default choice, no changes will be made to the states.
- ```NLAT1```
- ```NLAT2```
- ```NLAT3```

### Echo State Networks
Regarding ESNs in the library are implemented the following input layers:
- ```WeightedLayer```: weighted layer matrix with weights sampled from a uniform distribution.
- ```DenseLayer```: dense layer matrix with weights sampled from a uniform distribution.
- ```SparseLayer```: sparse layer matrix with weights sampled from a uniform distribution.
- ```MinimumLayer```: matrix with constant weights and weight sign decided following one of the two:
  - ```BernoulliSample```
  - ```IrrationalSample```
- ```InformedLayer```: special kin of weighted layer matrix for Hybrid ESNs.
 
The package also contains multiple implementation of Reservoirs:
- ```RandSparseReservoir```: random sparse matrix with scaling of spectral radius
- ```PseudoSVDReservoir```: Pseudo SVD construction of a random sparse matrix
- ```DelayLineReservoir```: minimal matrix with chosen weights
- ```DelayLineBackwardReservoir```: minimal matrix with chosen weights
- ```SimpleCycleReservoir```: minimal matrix with chosen weights
- ```CycleJumpsReservoir```: minimal matrix with chosen weights
 
In addition multiple ways of driving the reservoir states are also provided:
- ```RNN```: standard Recurrent Neural Network driver.
- ```MRNN```: Multiple RNN driver, it consists on a linear combination of RNNs
- ```GRU```: gated Recurrent Unit driver, with all the possible GRU variants available:
  - ```FullyGated```
  - ```Variant1```
  - ```Variant2```
  - ```Variant3```
  - ```Minimal```

An hybrid version of the model is also available through ```Hybrid```
