@doc raw"""
    RMNResESN(in_dims, mem_dims, res_dims, out_dims, activation=tanh;
        init_memory_reservoir=simple_cycle(; cycle_weight=1),
        init_memory_input=scaled_rand,
        init_memory_bias=zeros32,
        init_memory_state=randn32,
        use_memory_bias=False(),
        init_reservoir=rand_sparse,
        init_input=scaled_rand,
        init_memory=scaled_rand,
        init_orthogonal=orthogonal,
        init_bias=zeros32,
        init_state=randn32,
        use_bias=False(),
        alpha=1.0, beta=1.0,
        state_modifiers=(),
        readout_activation=identity)

Residual Reservoir Memory Network [Ceni2025b](@cite). Combines a linear memory
reservoir with a residual nonlinear reservoir that additionally consumes the
memory state, following the same composition pattern as [`RMNESN`](@ref) but
using a [`MemoryResESNCell`](@ref) as the nonlinear reservoir.

# Equations

```math
\begin{aligned}
    \mathbf{m}(t) &= \mathbf{W}_{\text{in}}^{m}\, \mathbf{u}(t)
        + \mathbf{C}\, \mathbf{m}(t-1) + \mathbf{b}^{m} \\
    \mathbf{h}(t) &= \alpha\, \mathbf{O}\, \mathbf{h}(t-1)
        + \beta\, \phi\!\left(\mathbf{W}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{W}_r\, \mathbf{h}(t-1)
        + \mathbf{W}_m\, \mathbf{m}(t-1)
        + \mathbf{b}\right)
\end{aligned}
```

# Arguments

  - `in_dims`: Input dimension.
  - `mem_dims`: Linear memory reservoir dimension.
  - `res_dims`: Nonlinear reservoir hidden state dimension.
  - `out_dims`: Output dimension.
  - `activation`: Activation function for the nonlinear reservoir. Default:
    `tanh`.

# Keyword arguments

  - `init_memory_reservoir`: Initializer for the memory reservoir recurrent
    matrix `C`. Default: `simple_cycle(; cycle_weight=1)`.
  - `init_memory_input`: Initializer for the memory reservoir input matrix.
    Default: [`scaled_rand`](@ref).
  - `init_memory_bias`: Initializer for the memory reservoir bias. Default:
    `zeros32`.
  - `init_memory_state`: Initializer used when an external memory state is not
    provided. Default: `randn32`.
  - `use_memory_bias`: Whether the memory reservoir uses a bias term. Default:
    `False()`.
  - `init_reservoir`: Initializer for the nonlinear reservoir recurrent matrix
    `W_r`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for the nonlinear reservoir input matrix `W_in`.
    Default: [`scaled_rand`](@ref).
  - `init_memory`: Initializer for the matrix coupling the memory state into
    the nonlinear reservoir `W_m`. Default: [`scaled_rand`](@ref).
  - `init_orthogonal`: Initializer for the orthogonal skip matrix `O`. Default:
    `orthogonal`.
  - `init_bias`: Initializer for the nonlinear reservoir bias, used iff
    `use_bias=true`. Default: `zeros32`.
  - `init_state`: Initializer used when an external nonlinear reservoir state
    is not provided. Default: `randn32`.
  - `use_bias`: Whether the nonlinear reservoir uses a bias term. Default:
    `False()`.
  - `alpha`: Residual skip weight `α`. Default: `1.0`.
  - `beta`: Nonlinear transform weight `β`. Default: `1.0`.
  - `state_modifiers`: A layer or collection of layers applied to the nonlinear
    reservoir state before the readout. Accepts a single layer, an
    `AbstractVector`, or a `Tuple`. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default:
    `identity`.

# Inputs

  - `x :: AbstractArray (in_dims, batch)`

# Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state `NamedTuple`.

# Parameters

  - `reservoir` — parameters of the internal [`RMNCell`](@ref), containing:
    - `linear_reservoir` — parameters of the memory [`ESNCell`](@ref),
      including:
      - `input_matrix :: (mem_dims × in_dims)`.
      - `reservoir_matrix :: (mem_dims × mem_dims)`.
      - `bias :: (mem_dims,)` — present only if `use_memory_bias=true`.
    - `nonlinear_reservoir` — parameters of [`MemoryResESNCell`](@ref),
      including:
      - `input_matrix :: (res_dims × in_dims)`.
      - `reservoir_matrix :: (res_dims × res_dims)`.
      - `memory_matrix :: (res_dims × mem_dims)`.
      - `orthogonal_matrix :: (res_dims × res_dims)`.
      - `bias :: (res_dims,)` — present only if `use_bias=true`.
  - `states_modifiers` — a `Tuple` with parameters for each modifier layer;
    may be empty.
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
    - `weight :: (out_dims × res_dims)`.
    - `bias :: (out_dims,)`.

# States

  - `reservoir` — states for the internal [`RMNCell`](@ref), containing:
    - `linear_reservoir` — states for the memory [`ESNCell`](@ref).
    - `nonlinear_reservoir` — states for [`MemoryResESNCell`](@ref).
    - `rng` — random number generator state used to sample initial recurrent
      states.
  - `states_modifiers` — a `Tuple` with states for each modifier layer.
  - `readout` — states for [`LinearReadout`](@ref).

# Reference

The original paper introduces this model under the name *Residual Reservoir
Memory Network* (ResRMN); see [Ceni2025b](@cite). In this package the type is
named `RMNResESN` to keep the RMN-family naming convention consistent with
[`RMNESN`](@ref).

# See also

[`RMNCell`](@ref), [`MemoryResESNCell`](@ref), [`ESNCell`](@ref),
[`LinearReadout`](@ref)
"""
@concrete struct RMNResESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function RMNResESN(
        in_dims::IntegerType, mem_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        init_memory_reservoir = simple_cycle(; cycle_weight = 1),
        init_memory_input = scaled_rand, init_memory_bias = zeros32,
        init_memory_state = randn32, use_memory_bias = False(),
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_memory = scaled_rand, init_orthogonal = orthogonal,
        init_bias = zeros32, init_state = randn32, use_bias = False(),
        alpha::AbstractFloat = 1.0, beta::AbstractFloat = 1.0,
        state_modifiers = (), readout_activation = identity
    )
    linear_reservoir = ESNCell(
        in_dims => mem_dims, identity;
        use_bias = use_memory_bias,
        init_bias = init_memory_bias,
        init_reservoir = init_memory_reservoir,
        init_input = init_memory_input,
        init_state = init_memory_state
    )
    nonlinear_reservoir = MemoryResESNCell(
        (in_dims, mem_dims) => res_dims, activation;
        use_bias = use_bias,
        init_bias = init_bias,
        init_reservoir = init_reservoir,
        init_input = init_input,
        init_memory = init_memory,
        init_orthogonal = init_orthogonal,
        init_state = init_state,
        alpha = alpha,
        beta = beta,
    )
    rmncell = RMNCell(nonlinear_reservoir, linear_reservoir)
    cell = StatefulLayer(rmncell)
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return RMNResESN(cell, mods, ro)
end

function Base.show(io::IO, esn::RMNResESN)
    print(io, "RMNResESN(\n")

    print(io, "    reservoir = ")
    show(io, esn.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(esn.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(esn.states_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, esn.readout)
    print(io, "\n)")

    return
end
