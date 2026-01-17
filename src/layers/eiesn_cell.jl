@doc raw"""
    EIESNCell(in_dims => out_dims, [activation]; kwargs...)

Excitatory-Inhibitory Echo State Network (EIESN) cell.
dynamics, inspired by biologically motivated excitation–inhibition balance.

This cell implements the state update rule corresponding to **Model 1** from
Issue #353, where the input is applied inside the nonlinearity:

```math
\mathbf{x}(t) =
b_{\mathrm{ex}} \, \phi\!\left(\mathbf{W}_{\mathrm{in}} \mathbf{u}(t) + a_{\mathrm{ex}} \mathbf{A} \mathbf{x}(t-1)\right)
- b_{\mathrm{inh}} \, \phi\!\left(\mathbf{W}_{\mathrm{in}} \mathbf{u}(t) + a_{\mathrm{inh}} \mathbf{A} \mathbf{x}(t-1)\right)
```

## Symbols

  - $\mathbf{x}(t)$: Reservoir state at time $t$.
  - $\mathbf{u}(t)$: Input at time $t$.
  - $\mathbf{A}$: Reservoir (recurrent) matrix.
  - $\mathbf{W}_{\mathrm{in}}$: Input matrix.
  - $a_{\mathrm{ex}}, a_{\mathrm{inh}}$: Excitatory and inhibitory recurrence scales.
  - $b_{\mathrm{ex}}, b_{\mathrm{inh}}$: Excitatory and inhibitory output scales.
  - $\phi$: Pointwise activation function.

The reservoir parameters are fixed after initialization; only the readout
layer is intended to be trained, following the standard reservoir computing paradigm.

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir (hidden state) dimension.
  - `activation`: Activation function $\phi$. Default: `tanh_fast`.

## Keyword arguments

  - `a_ex`: Excitatory recurrence scaling factor ($a_{\mathrm{ex}}$). Default: `0.9`.
  - `a_inh`: Inhibitory recurrence scaling factor ($a_{\mathrm{inh}}$). Default: `0.5`.
  - `b_ex`: Excitatory output scaling factor ($b_{\mathrm{ex}}$). Default: `1.0`.
  - `b_inh`: Inhibitory output scaling factor ($b_{\mathrm{inh}}$). Default: `1.0`.
  - `init_reservoir`: Initializer for the reservoir matrix $\mathbf{A}$. Default: `rand_sparse`.
  - `init_input`: Initializer for the input matrix $\mathbf{W}_{\mathrm{in}}$. Default: `scaled_rand`.
  - `init_state`: Initializer for the initial hidden state $\mathbf{x}(0)$. Default: `randn32`.

## Parameters

Created by `initialparameters(rng, cell)`:

  - `input_matrix :: (out_dims × in_dims)` — $\mathbf{W}_{\mathrm{in}}$.
  - `reservoir_matrix :: (out_dims × out_dims)` — $\mathbf{A}$.

## States

Created by `initialstates(rng, cell)`:

  - `rng`: Replicated RNG used to initialize the hidden state when an external
    state is not provided.

## Notes

  - This implementation corresponds to Model 1 described in Issue #353.
  - The structure allows for future extensions (e.g., additive-input variants).
"""

@concrete struct EIESNCell <: AbstractReservoirRecurrentCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    a_ex
    a_inh
    b_ex
    b_inh
    init_reservoir
    init_input
    init_state
    
end

function EIESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh_fast;
        a_ex = 0.9, a_inh = 0.5, b_ex = 1.0, b_inh = 1.0,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32
    )
    return EIESNCell(
        activation, in_dims, out_dims, a_ex, a_inh, b_ex,
        b_inh, init_reservoir, init_input, init_state
    )
end

function initialparameters(rng::AbstractRNG, cell::EIESNCell)
    ps = (
        input_matrix = cell.init_input(rng, cell.out_dims, cell.in_dims),
        reservoir_matrix = cell.init_reservoir(rng, cell.out_dims, cell.out_dims),
    )
    return ps
end

function initialstates(rng::AbstractRNG, cell::EIESNCell)
    return (rng = sample_replicate(rng),)
end

function (cell::EIESNCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, cell, inp)
    return cell((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (cell::EIESNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)
    win = ps.input_matrix
    A = ps.reservoir_matrix
    win_inp = dense_bias(win, inp, nothing)
    rec_ex  = cell.a_ex  .* (A * hidden_state)
    rec_inh = cell.a_inh .* (A * hidden_state)
    z_ex = win_inp .+ rec_ex 
    z_inh = win_inp .+ rec_inh
    h_ex = cell.activation.(z_ex)
    h_inh = cell.activation.(z_inh)
    h_new = cell.b_ex .* h_ex .- cell.b_inh .* h_inh
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, cell::EIESNCell)
    print(io, "EIESNCell($(cell.in_dims) => $(cell.out_dims)")
    cell.a_ex  != 0.9  && print(io, ", a_ex=$(cell.a_ex)")
    cell.a_inh != 0.5  && print(io, ", a_inh=$(cell.a_inh)")
    cell.b_ex  != 1.0  && print(io, ", b_ex=$(cell.b_ex)")
    cell.b_inh != 1.0  && print(io, ", b_inh=$(cell.b_inh)")
    return print(io, ")")
end