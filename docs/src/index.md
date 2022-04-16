# Overview

Reservoir Computing is an umbrella term used to describe a family of models such as Echo State Networks (ESNs) and Liquid State Machines (LSMs). The key concept is to expand the input data into a higher dimension and use regression in order to train the model; in some ways Reservoir Computers can be considered similar to kernel methods.

!!! info "Introductory material"
    This library assumes some basic knowledge of Reservoir Computing. For a good introduction, we suggest the following papers: the first two are the seminal papers about ESN and LSM, the others are in-depth review papers that should cover all the needed information.

    * Jaeger, Herbert: The “echo state” approach to analysing and training recurrent neural networks-with an erratum note.
    * Maass W, Natschläger T, Markram H: Real-time computing without stable states: a new framework for neural computation based on perturbations.
    * Lukoševičius, Mantas: A practical guide to applying echo state networks." Neural networks: Tricks of the trade.
    * Lukoševičius, Mantas, and Herbert Jaeger: Reservoir computing approaches to recurrent neural network training.

In this package (for the moment) there are the following models:
- Echo State Networks (ESNs)
- Support Vector Echo State Machines \[1\] (SVESMs)
- Echo State Gaussian Processes \[2\] (ESGPs)
- Reservoir Computing with Cellular Automata \[3\] (RECAs)
- Reservoir Memory Machine \[4\] (RMMs)
- Double Activation Echo State Networks \[5\] (DAFESNs)

Multiple features are present as well, like the possibility of using a number of different reservoir and input layer architectures, as well as different linear regression methods. For more information on this please refer to the examples.

# Installation
Since ReservoirComputing is registered in the Julia General Registry, it will suffice to do the following in the Julia REPL:
```
]add ReservoirComputing
```

## Citing

If you use this library in your work, please cite:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.05117,
  doi = {10.48550/ARXIV.2204.05117},
  url = {https://arxiv.org/abs/2204.05117},
  author = {Martinuzzi, Francesco and Rackauckas, Chris and Abdelrehim, Anas and Mahecha, Miguel D. and Mora, Karin},
  keywords = {Computational Engineering, Finance, and Science (cs.CE), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ReservoirComputing.jl: An Efficient and Modular Library for Reservoir Computing Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## References


[1]: Shi, Zhiwei, and Min Han. "Support vector echo-state machine for chaotic time-series prediction." IEEE Transactions on Neural Networks 18.2 (2007): 359-372.

[2]: Chatzis, Sotirios P., and Yiannis Demiris. "Echo state Gaussian process." IEEE Transactions on Neural Networks 22.9 (2011): 1435-1445.

[3]: Yilmaz, Ozgur. "Reservoir computing using cellular automata." arXiv preprint arXiv:1410.0162 (2014).

[4]: Paaßen, Benjamin, and Alexander Schulz. "Reservoir memory machines." arXiv preprint arXiv:2003.04793 (2020).

[5]: Lun, Shu-Xian, et al. "A novel model of leaky integrator echo state network for time-series prediction." Neurocomputing 159 (2015): 58-66.
