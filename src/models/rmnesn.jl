@doc raw"""
    RMNESN(in_dims, mem_dims, res_dims, out_dims, activation=tanh;
        init_memory_reservoir=simple_cycle(; cycle_weight=1),
        init_memory_input=scaled_rand,
        init_memory_bias=zeros32,
        init_memory_state=randn32,
        use_memory_bias=False(),
        init_reservoir=rand_sparse,
        init_input=scaled_rand,
        init_memory=scaled_rand,
        init_bias=zeros32,
        init_state=randn32,
        use_bias=False(),
        state_modifiers=(),
        readout_activation=identity)

Construct a Reservoir Memory Network Echo State Network [Gallicchio2024b](@cite).

# Arguments

  - `in_dims`: Input dimension.
  - `mem_dims`: Linear memory reservoir dimension.
  - `res_dims`: Nonlinear reservoir hidden state dimension.
  - `out_dims`: Output dimension.
  - `activation`: Activation function for the nonlinear reservoir. Default:
    `tanh`.

# Keyword arguments

  - `init_memory_reservoir`: Initializer for the memory reservoir recurrent
    matrix. Default: `simple_cycle(; cycle_weight=1)`.
  - `init_memory_input`: Initializer for the memory reservoir input matrix.
    Default: [`scaled_rand`](@ref).
  - `init_memory_bias`: Initializer for the memory reservoir bias. Default:
    `zeros32`.
  - `init_memory_state`: Initializer used when an external memory state is not
    provided. Default: `randn32`.
  - `use_memory_bias`: Whether the memory reservoir uses a bias term. Default:
    `False()`.
  - `init_reservoir`: Initializer for the nonlinear reservoir recurrent matrix.
    Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for the nonlinear reservoir input matrix. Default:
    [`scaled_rand`](@ref).
  - `init_memory`: Initializer for the matrix coupling the memory state into the
    nonlinear reservoir. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initializer for the nonlinear reservoir bias, used iff
    `use_bias=true`. Default: `zeros32`.
  - `init_state`: Initializer used when an external nonlinear reservoir state is
    not provided. Default: `randn32`.
  - `use_bias`: Whether the nonlinear reservoir uses a bias term. Default:
    `False()`.
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
    - `nonlinear_reservoir` — parameters of [`MemoryESNCell`](@ref),
      including:
      - `input_matrix :: (res_dims × in_dims)`.
      - `reservoir_matrix :: (res_dims × res_dims)`.
      - `memory_matrix :: (res_dims × mem_dims)`.
      - `bias :: (res_dims,)` — present only if `use_bias=true`.
  - `states_modifiers` — a `Tuple` with parameters for each modifier layer;
    may be empty.
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
    - `weight :: (out_dims × res_dims)`.
    - `bias :: (out_dims,)`.

# States

  - `reservoir` — states for the internal [`RMNCell`](@ref), containing:
    - `linear_reservoir` — states for the memory [`ESNCell`](@ref).
    - `nonlinear_reservoir` — states for [`MemoryESNCell`](@ref).
    - `rng` — random number generator state used to sample initial recurrent
      states.
  - `states_modifiers` — a `Tuple` with states for each modifier layer.
  - `readout` — states for [`LinearReadout`](@ref).

# See also

[`RMNCell`](@ref), [`MemoryESNCell`](@ref), [`ESNCell`](@ref),
[`LinearReadout`](@ref)

"""
@concrete struct RMNESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function RMNESN(
        in_dims::IntegerType, mem_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        init_memory_reservoir=simple_cycle(; cycle_weight=1),
        init_memory_input=scaled_rand, init_memory_bias=zeros32,
        init_memory_state=randn32, use_memory_bias=False(),
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_memory=scaled_rand, init_orthogonal=orthogonal,
        init_bias=zeros32, init_state=randn32, use_bias=False(),
        state_modifiers=(), readout_activation=identity
    )
    linear_reservoir = ESNCell(in_dims => mem_dims, identity;
        use_bias = use_memory_bias,
        init_bias = init_memory_bias,
        init_reservoir = init_memory_reservoir,
        init_input = init_memory_input,
        init_state = init_memory_state
    )
    nonlinear_reservoir = MemoryESNCell((in_dims, mem_dims) => res_dims;
        use_bias = use_bias,
        init_bias = init_bias,
        init_reservoir = init_reservoir,
        init_input = init_input,
        init_memory = init_memory,
        init_state = init_state,
    )
    rmncell = RMNCell(nonlinear_reservoir, linear_reservoir)
    cell = StatefulLayer(rmncell)
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return RMNESN(cell, mods, ro)
end
