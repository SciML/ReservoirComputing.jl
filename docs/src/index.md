# ReservoirComputing.jl

ReservoirComputing.jl provides an efficient, modular and easy to use implementation of Reservoir Computing models such as Echo State Networks (ESNs). Reservoir Computing (RC) is an umbrella term used to describe a family of models such as ESNs and Liquid State Machines (LSMs). The key concept is to expand the input data into a higher dimension and use regression in order to train the model; in some ways Reservoir Computers can be considered similar to kernel methods. 


!!! info "Introductory material"
    This library assumes some basic knowledge of Reservoir Computing. For a good introduction, we suggest the following papers: the first two are the seminal papers about ESN and LSM, the others are in-depth review papers that should cover all the needed information. For the majority of the algorithms implemented in this library we cited in the documentation the original work introducing them. If you ever are in doubt about about a method or a function just type ```? function``` in the Julia REPL to read the relevant notes.

    * Jaeger, Herbert: The “echo state” approach to analyzing and training recurrent neural networks-with an erratum note.
    * Maass W, Natschläger T, Markram H: Real-time computing without stable states: a new framework for neural computation based on perturbations.
    * Lukoševičius, Mantas: A practical guide to applying echo state networks." Neural networks: Tricks of the trade.
    * Lukoševičius, Mantas, and Herbert Jaeger: Reservoir computing approaches to recurrent neural network training.
    
!!! info "Performance tip"
    For faster computations on the CPU it is suggested to add `using MKL` to the script. For clarity's sake this library will not be indicated under every example in the documentation.
## Installation
ReservoirComputing.jl is registered in the General Julia Registry, so the installation of the package follows the usual procedure:
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
- ```PaddedStates```: the states are padded using a vertical concatenation with the choosing padding value
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

### Reservoir Computing with Cellular Automata
The package provides also an implementation of Reservoir Computing models based on one dimensional Cellular Automata through the ```RECA``` call. For the moment the only input encoding available (an input encoding plays a similar role to the input matrix for ESNs) is a random mapping, called through ```RandomMapping```. 

All the training methods described above can be used, as well as all the modifications to the states. Both prediction methods are also possible in theory, although in the literature only ```Predictive``` tasks have been explored.

### Contributing
Contributions are very welcomed! Some interesting variation of RC models are posted in the issues, but everyone is free to just post relevant papers that could fit the scope of the library. Help with the documentation, providing new examples or application cases is also really important and appreciated. Everything that can make the package a little better is a great contribution, no matter how small. The API section of the documentation provides a more in depth look into how things work and are connected, so that is a good place to start exploring more the library. For every doubt that cannot be expressed in issues please feel free to contact any of the lead developers on Slack or by email.

## Citing

If you use this library in your work, please cite:

```bibtex
@article{JMLR:v23:22-0611,
  author  = {Francesco Martinuzzi and Chris Rackauckas and Anas Abdelrehim and Miguel D. Mahecha and Karin Mora},
  title   = {ReservoirComputing.jl: An Efficient and Modular Library for Reservoir Computing Models},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {288},
  pages   = {1--8},
  url     = {http://jmlr.org/papers/v23/22-0611.html}
}
```

## Reproducibility
```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```
```@example
using Pkg # hide
Pkg.status() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>and using this machine and Julia version.</summary>
```
```@example
using InteractiveUtils # hide
versioninfo() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```
```@example
using Pkg # hide
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the 
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml"
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml"
```
```@raw html
">project</a> file.
```
