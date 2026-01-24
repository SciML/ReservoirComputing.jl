@doc raw"""
    InputDelayESN(in_dims, res_dims, out_dims, activation=tanh;
                  num_delays=1, stride=1, leak_coefficient=1.0,
                  init_reservoir=rand_sparse, init_input=scaled_rand,
                  init_bias=zeros32, init_state=randn32, use_bias=false,
                  input_modifiers=(), readout_activation=identity)

Echo State Network with input delays [Fleddermann2025](@cite).

`InputDelayESN` composes:
  1) a [`DelayLayer`](@ref) applied to the input signal to build
     tapped-delay features,
  2) zero or more additional `input_modifiers` applied after the delay,
  3) a stateful [`ESNCell`](@ref) (reservoir) receiving the augmented input,
     and
  4) a [`LinearReadout`](@ref) mapping the reservoir state to outputs.

At each time step, the input `u(t)` is expanded into a delay-coordinate
vector that stacks the current and `num_delays` past inputs. This augmented
signal is then used to update the reservoir state `x(t)`.

## Equations

```math
\begin{aligned}
    \mathbf{u}_{\mathrm{d}}(t) &= \begin{bmatrix} \mathbf{u}(t) \\
    \mathbf{u}(t-s) \\
    \vdots \\
    \mathbf{u}\!\bigl(t-Ds\bigr) \end{bmatrix},
        \qquad D=\text{num\_delays},\ \ s=\text{stride}, \\
    \mathbf{z}(t) &= \mathrm{Mods}\!\left(\mathbf{u}_{\mathrm{d}}(t)\right), \\
    \mathbf{x}(t) &= (1-\alpha)\, \mathbf{x}(t-1) + \alpha\, \phi\!\left(
        \mathbf{W}_{\text{in}}\, \mathbf{z}(t) + \mathbf{W}_r\,
        \mathbf{x}(t-1) + \mathbf{b} \right), \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{\text{out}}\,
        \mathbf{x}(t) + \mathbf{b}_{\text{out}} \right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation (for [`ESNCell`](@ref)). Default:
    `tanh`.

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

Delay expansion (on input):

  - `num_delays`: Number of past input states to include in the tapped-delay
    vector. The `DelayLayer` output has `(num_delays + 1) * in_dims` entries.
    Default: `1`.
  - `stride`: Delay stride in layer calls. The delay buffer is updated only
    when the internal clock is a multiple of `stride`. Default: `1`.

Composition:

  - `input_modifiers`: A layer or collection of layers applied to the delayed
    input features before entering the reservoir. Accepts a single layer,
    an `AbstractVector`, or a `Tuple`. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default:
    `identity`.

## Inputs

  - `x :: AbstractArray (in_dims, batch)`

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `input_modifiers` — a `Tuple` with parameters for:
      1. the internal [`DelayLayer`](@ref), and
      2. any user-provided modifier layers (may be empty).
  - `reservoir` — parameters of the internal [`ESNCell`](@ref), including:
      - `input_matrix :: (res_dims × ((num_delays + 1) * in_dims))` — `W_in`
      - `reservoir_matrix :: (res_dims × res_dims)` — `W_res`
      - `bias :: (res_dims,)` — present only if `use_bias=true`
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × res_dims)` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

## States

  - `input_modifiers` — a `Tuple` with states for the internal `DelayLayer`
    (its delay buffer and clock) and each additional modifier layer.
  - `reservoir` — states for the internal [`ESNCell`](@ref) (e.g. `rng`).
  - `readout` — states for [`LinearReadout`](@ref) (typically empty).
"""
@concrete struct InputDelayESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    input_modifiers
    reservoir
    readout
end

function InputDelayESN(
        in_dims::IntegerType, res_dims::Int, out_dims::IntegerType, activation = tanh;
        num_delays::Int = 2, stride::Int = 1, readout_activation = identity,
        input_modifiers = (), kwargs...
    )
    delay = DelayLayer(in_dims; num_delays = num_delays, stride = stride)
    input_mods_tuple = input_modifiers isa Tuple || input_modifiers isa AbstractVector ?
        (delay, input_modifiers...) : (delay, input_modifiers)
    input_mods = _wrap_layers(input_mods_tuple)
    augmented_in_dims = in_dims * (num_delays + 1)
    cell = StatefulLayer(ESNCell(augmented_in_dims => res_dims, activation; kwargs...))
    ro = LinearReadout(res_dims => out_dims, readout_activation)

    return InputDelayESN(input_mods, cell, ro)
end

function Base.show(io::IO, esn::InputDelayESN)
    print(io, "InputDelayESN(\n")

    print(io, "    input_modifiers = ")
    if isempty(esn.input_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(esn.input_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    reservoir = ")
    show(io, esn.reservoir)
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, esn.readout)
    print(io, "\n)")

    return
end
