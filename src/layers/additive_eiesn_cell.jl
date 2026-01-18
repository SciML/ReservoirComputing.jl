@doc raw"""
        AdditiveEIESNCell(in_dims => out_dims, [activation]; kwargs...)

Excitatory-Inhibitory Echo State Network (EIESN) cell with Additive Input [Panahi2025](@cite).

```math
\mathbf{x}(t) =
b_{\mathrm{ex}} \, \phi\!\left(a_{\mathrm{ex}} \mathbf{A} \mathbf{x}(t-1)\right)
- b_{\mathrm{inh}} \, \phi\!\left(a_{\mathrm{inh}} \mathbf{A} \mathbf{x}(t-1)\right)
+ g \, \mathbf{W}_{\mathrm{in}} \mathbf{u}(t)
```

## Symbols

    - $\mathbf{x}(t)$: Reservoir state at time $t$.
    - $\mathbf{u}(t)$: Input at time $t$.
    - $\mathbf{A}$: Reservoir (recurrent) matrix.
    - $\mathbf{W}_{\mathrm{in}}$: Input matrix.
    - $a_{\mathrm{ex}}, a_{\mathrm{inh}}$: Excitatory and inhibitory recurrence scales.
    - $b_{\mathrm{ex}}, b_{\mathrm{inh}}$: Excitatory and inhibitory output scales.
    - $g$: Input scaling factor.
    - $\phi$: Pointwise activation function.

The reservoir parameters are fixed after initialization; only the readout
layer is intended to be trained, following the standard reservoir computing paradigm.

## Arguments

    - `in_dims`: Input dimension.
    - `out_dims`: Reservoir (hidden state) dimension.
    - `activation`: Activation function $\phi$. Default: `tanh_fast`.

## Keyword arguments

    - `exc_recurrence_scale`: Excitatory recurrence scaling factor ($a_{\mathrm{ex}}$). Default: `0.9`.
    - `inh_recurrence_scale`: Inhibitory recurrence scaling factor ($a_{\mathrm{inh}}$). Default: `0.5`.
    - `exc_output_scale`: Excitatory output scaling factor ($b_{\mathrm{ex}}$). Default: `1.0`.
    - `inh_output_scale`: Inhibitory output scaling factor ($b_{\mathrm{inh}}$). Default: `1.0`.
    - `input_scale`: Input scaling factor ($g$). Default: `1.0`.
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
"""

@concrete struct AdditiveEIESNCell <: AbstractReservoirRecurrentCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    exc_recurrence_scale
    inh_recurrence_scale
    exc_output_scale
    inh_output_scale
    input_scale
    init_reservoir
    init_input
    init_state

end

function AdditiveEIESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh_fast;
        exc_recurrence_scale = 0.9, inh_recurrence_scale = 0.5,
        exc_output_scale = 1.0, inh_output_scale = 1.0,
        input_scale = 1.0,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32
    )
    return AdditiveEIESNCell(
        activation, in_dims, out_dims,
        exc_recurrence_scale, inh_recurrence_scale,
        exc_output_scale, inh_output_scale, input_scale,
        init_reservoir, init_input, init_state
    )
end

function initialparameters(rng::AbstractRNG, cell::AdditiveEIESNCell)
    ps = (
        input_matrix = cell.init_input(rng, cell.out_dims, cell.in_dims),
        reservoir_matrix = cell.init_reservoir(rng, cell.out_dims, cell.out_dims),
    )
    return ps
end

function initialstates(rng::AbstractRNG, cell::AdditiveEIESNCell)
    return (rng = sample_replicate(rng),)
end

function (cell::AdditiveEIESNCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, cell, inp)
    return cell((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (cell::AdditiveEIESNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)

    win = ps.input_matrix
    A = ps.reservoir_matrix
    win_inp = dense_bias(win, inp, nothing)
    rec_ex = T(cell.exc_recurrence_scale) .* (A * hidden_state)
    rec_inh = T(cell.inh_recurrence_scale) .* (A * hidden_state)
    h_ex = cell.activation.(rec_ex)
    h_inh = cell.activation.(rec_inh)
    reservoir_part = T(cell.exc_output_scale) .* h_ex .- T(cell.inh_output_scale) .* h_inh
    h_new = reservoir_part .+ (T(cell.input_scale) .* win_inp)

    return (h_new, (h_new,)), st
end

function Base.show(io::IO, cell::AdditiveEIESNCell)
    print(io, "AdditiveEIESNCell($(cell.in_dims) => $(cell.out_dims)")
    cell.exc_recurrence_scale != 0.9  && print(io, ", exc_recurrence_scale=$(cell.exc_recurrence_scale)")
    cell.inh_recurrence_scale != 0.5  && print(io, ", inh_recurrence_scale=$(cell.inh_recurrence_scale)")
    cell.exc_output_scale != 1.0  && print(io, ", exc_output_scale=$(cell.exc_output_scale)")
    cell.inh_output_scale != 1.0  && print(io, ", inh_output_scale=$(cell.inh_output_scale)")
    cell.input_scale != 1.0  && print(io, ", input_scale=$(cell.input_scale)")
    return print(io, ")")
end
