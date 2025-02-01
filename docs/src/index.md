# ReservoirComputing.jl

ReservoirComputing.jl is a versatile and user-friendly Julia package designed
for the implementation of Reservoir Computing models, such as Echo State Networks (ESNs).
Reservoir Computing expands the input data into a higher-dimensional
space, leveraging regression techniques for effective model training.
This approach can be thought as a kernel method with an explicit kernel trick.

!!! info "Introductory material"
    
    This library assumes some basic knowledge of Reservoir Computing.
    For a good introduction, we suggest the following papers:
    the first two are the seminal papers about ESN and liquid state machines,
    the others are in-depth review papers that should cover all the needed
    information. For the majority of the algorithms implemented in this library
    we cited in the documentation the original work introducing them.
    If you ever are in doubt about a method or a function just type `? function`
    in the Julia REPL to read the relevant notes.
    
      - Jaeger, Herbert: The “echo state” approach to analyzing and training
          recurrent neural networks-with an erratum note.
      - Maass W, Natschläger T, Markram H: Real-time computing without
          stable states: a new framework for neural computation based on
          perturbations.
      - Lukoševičius, Mantas: A practical guide to applying echo state networks.
          Neural networks: Tricks of the trade.
      - Lukoševičius, Mantas, and Herbert Jaeger: Reservoir computing approaches
          to recurrent neural network training.

!!! info "Performance tip"
    
    For faster computations on the CPU it is suggested to add `using MKL`
    to the script. For clarity's sake this library will not be indicated 
    under every example in the documentation.

## Installation

To install ReservoirComputing.jl, ensure you have Julia version 1.10 or higher.
Follow these steps:

    1. Open the Julia command line.
    2. Enter the Pkg REPL mode by pressing ].
    3. Type `add ReservoirComputing` and press Enter.

For a more customized installation or to contribute to the package,
consider cloning the repository:

```julia
using Pkg
Pkg.clone("https://github.com/SciML/ReservoirComputing.jl.git")
```

or `dev` the package.

## Features Overview

  - **Multiple Training Algorithms**: Supports Ridge Regression, Linear Models,
      and LIBSVM regression methods for Reservoir Computing models.
  - **Diverse Prediction Methods**: Offers both generative and predictive methods
      for Reservoir Computing predictions.
  - **Modifiable Training and Prediction**: Allows modifications in Reservoir
      Computing states, such as state extension, padding, and combination methods.
  - **Non-linear Algorithm Options**: Includes options for non-linear
      modifications in algorithms.
  - **Echo State Networks (ESNs)**: Features various input layers, reservoirs,
      and methods for driving ESN reservoir states.
  - **Cellular Automata-Based Reservoir Computing**: Introduces models based
      on one-dimensional Cellular Automata for Reservoir Computing.

## Contributing

Contributions to ReservoirComputing.jl are highly encouraged and appreciated.
Whether it's through implementing new RC model variations,
enhancing documentation, adding examples, or any improvement,
your contribution is valuable.
We welcome posts of relevant papers or ideas in the issues section.
For deeper insights into the library's functionality, the API section in the
documentation is a great resource. For any queries not suited for issues,
please reach out to the lead developers via Slack or email.

## Citing

If you use ReservoirComputing.jl in your work, we kindly ask you to cite it.
Here is the BibTeX entry for your convenience:

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
Pkg.status(; mode=PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
