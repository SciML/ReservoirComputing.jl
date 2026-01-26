@doc raw"""
    FullDelayESN(in_dims, res_dims, out_dims, activation=tanh;
                 num_input_delays=1, input_stride=1,
                 num_state_delays=1, state_stride=1,
                 leak_coefficient=1.0, init_reservoir=rand_sparse,
                 init_input=scaled_rand, init_bias=zeros32,
                 init_state=randn32, use_bias=false,
                 states_modifiers=(), readout_activation=identity)

Echo State Network with both input and state delays [Fleddermann2025](@cite).

`FullDelayESN` composes:
  1) an internal [`DelayLayer`](@ref) applied to the input signal,
  2) a stateful [`ESNCell`](@ref) (reservoir) receiving the augmented input,
  3) a second internal [`DelayLayer`](@ref) applied to the reservoir state,
  4) zero or more additional `states_modifiers` applied after the state
     delay, and
  5) a [`LinearReadout`](@ref) mapping the final feature vector to outputs.

At each time step, the input `u(t)` is expanded into a delay-coordinate
vector `u_d(t)`. This drives the reservoir to produce state `x(t)`. Finally,
`x(t)` is expanded into a state delay-coordinate vector `x_d(t)` before
passing to modifiers and readout.

## Equations

```math
\begin{aligned}
    \mathbf{u}_{\mathrm{d}}(t) &= \begin{bmatrix} \mathbf{u}(t) \\
    \mathbf{u}(t-s_{in}) \\
    \vdots \\
    \mathbf{u}\!\bigl(t-D_{in}s_{in}\bigr) \end{bmatrix},
        \qquad D_{in}=\text{num\_input\_delays},\ \
        s_{in}=\text{input\_stride}, \\
    \mathbf{x}(t) &= (1-\alpha)\, \mathbf{x}(t-1) + \alpha\, \phi\!\left(
        \mathbf{W}_{\text{in}}\, \mathbf{u}_{\mathrm{d}}(t) +
        \mathbf{W}_r\, \mathbf{x}(t-1) + \mathbf{b} \right), \\
    \mathbf{x}_{\mathrm{d}}(t) &= \begin{bmatrix} \mathbf{x}(t) \\
    \mathbf{x}(t-s_{st}) \\
    \vdots \\
    \mathbf{x}\!\bigl(t-D_{st}s_{st}\bigr) \end{bmatrix},
        \qquad D_{st}=\text{num\_state\_delays},\ \
        s_{st}=\text{state\_stride}, \\
    \mathbf{z}(t) &= \mathrm{Mods}\!\left(
        \mathbf{x}_{\mathrm{d}}(t)\right), \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{\text{out}}\,
        \mathbf{z}(t) + \mathbf{b}_{\text{out}} \right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation (for [`ESNCell`](@ref)).
    Default: `tanh`.

## Keyword arguments

Reservoir (passed to [`ESNCell`](@ref)):

  - `leak_coefficient`: Leak rate in `(0, 1]`. Default: `1.0`.
  - `init_reservoir`: Initializer for `W_res`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initializer for reservoir bias (used iff `use_bias=true`).
    Default: `zeros32`.
  - `init_state`: Initializer used when an external state is not provided.
    Default: `randn32`.
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.

Input delay expansion:

  - `num_input_delays`: Number of past input steps to include. The effective
    input to the reservoir has size `(num_input_delays + 1) * in_dims`.
    Default: `1`.
  - `input_stride`: Stride for the input delay buffer. Default: `1`.

State delay expansion:

  - `num_state_delays`: Number of past reservoir states to include. The
    readout receives a vector of size `(num_state_delays + 1) * res_dims`.
    Default: `1`.
  - `state_stride`: Stride for the state delay buffer. Default: `1`.

Composition:

  - `states_modifiers`: A layer or collection of layers applied to the
    delayed reservoir features before the readout. These run **after** the
    internal state [`DelayLayer`](@ref). Accepts a single layer, an
    `AbstractVector`, or a `Tuple`. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout.
    Default: `identity`.

## Inputs

  - `x :: AbstractArray (in_dims, batch)`

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `input_delay` — parameters for the input [`DelayLayer`](@ref).
  - `reservoir` — parameters of the internal [`ESNCell`](@ref), including:
      - `input_matrix :: (res_dims × ((num_input_delays + 1) * in_dims))`
        — `W_in`
      - `reservoir_matrix :: (res_dims × res_dims)` — `W_res`
      - `bias :: (res_dims,)` — present only if `use_bias=true`
  - `state_delay` — parameters for the state [`DelayLayer`](@ref).
  - `states_modifiers` — parameters for the user-provided modifier layers.
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × ((num_state_delays + 1) * res_dims))`
        — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

## States

  - `input_delay` — state for the input [`DelayLayer`](@ref) (buffer and
    clock).
  - `reservoir` — states for the internal [`ESNCell`](@ref) (e.g. `rng`
    used to sample initial hidden states).
  - `state_delay` — state for the state [`DelayLayer`](@ref) (buffer and
    clock).
  - `states_modifiers` — states for the user-provided modifier layers.
  - `readout` — states for [`LinearReadout`](@ref) (typically empty).
"""

@concrete struct FullDelayESN <:
    AbstractEchoStateNetwork{
        (
            :input_delay, :reservoir, :state_delay,
            :states_modifiers, :readout,
        ),
    }
    input_delay
    reservoir
    state_delay
    states_modifiers
    readout
end

function FullDelayESN(
        in_dims::IntegerType,
        res_dims::Int, out_dims::IntegerType, activation = tanh;
        num_input_delays::Int = 1, input_stride::Int = 1, num_state_delays::Int = 1,
        state_stride::Int = 1, readout_activation = identity, states_modifiers = (),
        kwargs...
    )

    input_delay = DelayLayer(in_dims; num_delays = num_input_delays, stride = input_stride)
    augmented_in_dims = in_dims * (num_input_delays + 1)
    cell = StatefulLayer(ESNCell(augmented_in_dims => res_dims, activation; kwargs...))
    state_delay = DelayLayer(res_dims; num_delays = num_state_delays, stride = state_stride)
    augmented_res_dims = res_dims * (num_state_delays + 1)
    mods_tuple = states_modifiers isa Tuple || states_modifiers isa AbstractVector ?
        Tuple(states_modifiers) : (states_modifiers,)
    st_mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(augmented_res_dims => out_dims, readout_activation)
    return FullDelayESN(input_delay, cell, state_delay, st_mods, ro)
end

function Base.show(io::IO, esn::FullDelayESN)
    print(io, "FullDelayESN(\n")

    print(io, "    input_delay = ")
    show(io, esn.input_delay)
    print(io, ",\n")

    print(io, "    reservoir = ")
    show(io, esn.reservoir)
    print(io, ",\n")

    print(io, "    state_delay = ")
    show(io, esn.state_delay)
    print(io, ",\n")

    print(io, "    states_modifiers = ")
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
