# Altering States

In ReservoirComputing models, it's possible to perform alterations on the reservoir states during the training stage. These alterations can improve prediction results or replicate results found in the literature. Alterations are categorized into two possibilities: padding or extending the states, and applying non-linear algorithms to the states.

## Padding and Extending States

### Extending States

Extending the states involves appending the corresponding input values to the reservoir states. If $\textbf{x}(t)$ represents the reservoir state at time $t$ corresponding to the input $\textbf{u}(t)$, the extended state is represented as $[\textbf{x}(t); \textbf{u}(t)]$, where $[;]$ denotes vertical concatenation. This procedure is commonly used in Echo State Networks. You can extend the states in every ReservoirComputing.jl model by using the `states_type` keyword argument and calling the `ExtendedStates()` method. No additional arguments are needed.

### Padding States

Padding the states involves appending a constant value, such as 1.0, to each state. In the notation introduced earlier, padded states can be represented as $[\textbf{x}(t); 1.0]$. This approach is detailed in "A practical guide to applying echo state networks." by Lukoševičius, Mantas. To pad the states, you can use the `states_type` keyword argument and call the `PaddedStates(padding)` method, where `padding` represents the value to be concatenated to the states. By default, the padding value is set to 1.0, so most of the time, calling `PaddedStates()` will suffice.

Additionally, you can pad the extended states by using the `PaddedExtendedStates(padding)` method, which also has a default padding value of 1.0.

You can choose not to apply any of these changes to the states by calling `StandardStates()`, which is the default choice for the states.

## Non-Linear Algorithms

First introduced in [Pathak2017](@cite) and expanded in [Chattopadhyay2020](@cite), non-linear algorithms are nonlinear combinations of the columns of the matrix states. There are three such algorithms implemented in ReservoirComputing.jl, and you can choose which one to use with the `nla_type` keyword argument. The default value is set to `NLADefault()`, which means no non-linear algorithm is applied.

The available non-linear algorithms are:

  - `NLAT1()`
  - `NLAT2()`
  - `NLAT3()`

These algorithms perform specific operations on the reservoir states. To provide a better understanding of what they do, let ``\textbf{x}_{i, j}`` be elements of the state matrix, with ``i=1,...,T \ j=1,...,N`` where ``T`` is the length of the training and ``N`` is the reservoir size.

**NLAT1**

```math
\tilde{\textbf{x}}_{i,j} = \textbf{x}_{i,j} \times \textbf{x}_{i,j} \ \ \text{if j is odd} \\
\tilde{\textbf{x}}_{i,j} = \textbf{x}_{i,j}  \ \ \text{if j is even}
```

**NLAT2**

```math
\tilde{\textbf{x}}_{i,j} = \textbf{x}_{i,j-1} \times \textbf{x}_{i,j-2} \ \ \text{if j > 1 is odd} \\
\tilde{\textbf{x}}_{i,j} = \textbf{x}_{i,j}  \ \ \text{if j is 1 or even}
```

**NLAT3**

```math
\tilde{\textbf{x}}_{i,j} = \textbf{x}_{i,j-1} \times \textbf{x}_{i,j+1} \ \ \text{if j > 1 is odd} \\
\tilde{\textbf{x}}_{i,j} = \textbf{x}_{i,j}  \ \ \text{if j is 1 or even}
```

## References

```@bibliography
Pages = ["states_variation.md"]
Canonical = false
```