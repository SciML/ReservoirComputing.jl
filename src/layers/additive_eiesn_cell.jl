@doc raw"""
    AdditiveEIESNCell(in_dims => out_dims, [activation]; kwargs...)

Excitatory-Inhibitory Echo State Network (EIESN) cell with additive input
[Panahi2025](@cite).

```math
\mathbf{x}(t) =
b_{\mathrm{ex}} \, \phi_{\mathrm{ex}}\!\left(
  a_{\mathrm{ex}} \mathbf{A} \mathbf{x}(t-1) + \mathbf{\beta}_{\mathrm{ex}}\right)
- b_{\mathrm{inh}} \, \phi_{\mathrm{inh}}\!\left(
  a_{\mathrm{inh}} \mathbf{A} \mathbf{x}(t-1) + \mathbf{\beta}_{\mathrm{inh}}\right)
+ g\!\left(\mathbf{W}_{\mathrm{in}} \mathbf{u}(t) + \mathbf{\beta}_{\mathrm{in}}\right)
```

## Symbols

  - $\mathbf{x}(t)$: Reservoir state at time $t$.
  - $\mathbf{u}(t)$: Input at time $t$.
  - $\mathbf{A}$: Reservoir (recurrent) matrix.
  - $\mathbf{W}_{\mathrm{in}}$: Input matrix.
  - $\mathbf{\beta}_{\mathrm{ex}}, \mathbf{\beta}_{\mathrm{inh}},
    \mathbf{\beta}_{\mathrm{in}}$: Bias vectors (optional).
  - $a_{\mathrm{ex}}, a_{\mathrm{inh}}$: Excitatory and inhibitory recurrence scales.
  - $b_{\mathrm{ex}}, b_{\mathrm{inh}}$: Excitatory and inhibitory output scales.
  - $g$: Input activation function.
  - $\phi_{\mathrm{ex}}, \phi_{\mathrm{inh}}$: Excitatory and inhibitory activation
    functions.

The reservoir parameters are fixed after initialization; only the readout
layer is intended to be trained, following the standard reservoir computing paradigm.

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir (hidden state) dimension.
  - `activation`: Activation function. Can be a single function (applied to
    both terms) or a `Tuple` of two functions $(\phi_{\mathrm{ex}},
    \phi_{\mathrm{inh}})$. Default: `tanh_fast`.

## Keyword arguments

  - `input_activation`: The non-linear function $g$ applied to the input
    term. Default: `identity`.
  - `use_bias`: Boolean to enable/disable bias vectors ($\mathbf{\beta}$).
    Default: `true`.
  - `exc_recurrence_scale`: Excitatory recurrence scaling factor
    ($a_{\mathrm{ex}}$). Default: `0.9`.
  - `inh_recurrence_scale`: Inhibitory recurrence scaling factor
    ($a_{\mathrm{inh}}$). Default: `0.5`.
  - `exc_output_scale`: Excitatory output scaling factor ($b_{\mathrm{ex}}$).
    Default: `1.0`.
  - `inh_output_scale`: Inhibitory output scaling factor ($b_{\mathrm{inh}}$).
    Default: `1.0`.
  - `init_reservoir`: Initializer for the reservoir matrix $\mathbf{A}$.
    Default: `rand_sparse`.
  - `init_input`: Initializer for the input matrix $\mathbf{W}_{\mathrm{in}}$.
    Default: `scaled_rand`.
  - `init_bias`: Initializer for the bias vectors. Default: `zeros32`.
  - `init_state`: Initializer for the initial hidden state $\mathbf{x}(0)$.
    Default: `randn32`.

## Parameters

Created by `initialparameters(rng, cell)`:

  - `input_matrix :: (out_dims × in_dims)` — $\mathbf{W}_{\mathrm{in}}$.
  - `reservoir_matrix :: (out_dims × out_dims)` — $\mathbf{A}$.
  - `bias_ex`, `bias_inh`, `bias_in` — Bias vectors (present only if
    `use_bias=true`).

## States

Created by `initialstates(rng, cell)`:

  - `rng`: Replicated RNG used to initialize the hidden state when an external
    state is not provided.
"""
@concrete struct AdditiveEIESNCell <: AbstractEchoStateNetworkCell
    activation <: Tuple
    input_activation
    use_bias <: StaticBool
    in_dims <: IntegerType
    out_dims <: IntegerType
    exc_recurrence_scale
    inh_recurrence_scale
    exc_output_scale
    inh_output_scale
    init_reservoir
    init_input
    init_bias
    init_state

end

function AdditiveEIESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh_fast;
        input_activation = identity,
        use_bias::BoolType = True(),
        exc_recurrence_scale = 0.9, inh_recurrence_scale = 0.5,
        exc_output_scale = 1.0, inh_output_scale = 1.0,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_bias = zeros32,
        init_state = randn32
    )
    activation isa Tuple || (activation = ntuple(Returns(activation), 2))
    return AdditiveEIESNCell(
        activation, input_activation, static(use_bias), in_dims, out_dims,
        exc_recurrence_scale, inh_recurrence_scale,
        exc_output_scale, inh_output_scale,
        init_reservoir, init_input, init_bias, init_state
    )
end

function initialparameters(rng::AbstractRNG, cell::AdditiveEIESNCell)
    ps = (
        input_matrix = cell.init_input(rng, cell.out_dims, cell.in_dims),
        reservoir_matrix = cell.init_reservoir(rng, cell.out_dims, cell.out_dims),
    )
    if known(cell.use_bias)
        ps = merge(
            ps, (
                bias_ex = cell.init_bias(rng, cell.out_dims),
                bias_inh = cell.init_bias(rng, cell.out_dims),
                bias_in = cell.init_bias(rng, cell.out_dims),
            )
        )
    end
    return ps
end

function (cell::AdditiveEIESNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)

    b_ex = safe_getproperty(ps, Val(:bias_ex))
    b_inh = safe_getproperty(ps, Val(:bias_inh))
    b_in = safe_getproperty(ps, Val(:bias_in))
    win_out = dense_bias(ps.input_matrix, inp, b_in)
    g_out = cell.input_activation.(win_out)
    Ax = ps.reservoir_matrix * hidden_state
    arg_ex = T(cell.exc_recurrence_scale) .* Ax
    if b_ex !== nothing
        arg_ex = arg_ex .+ b_ex
    end
    h_ex = cell.activation[1].(arg_ex)
    arg_inh = T(cell.inh_recurrence_scale) .* Ax
    if b_inh !== nothing
        arg_inh = arg_inh .+ b_inh
    end
    h_inh = cell.activation[2].(arg_inh)
    h_new = (T(cell.exc_output_scale) .* h_ex) .- (T(cell.inh_output_scale) .* h_inh) .+ g_out
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, cell::AdditiveEIESNCell)
    print(io, "AdditiveEIESNCell($(cell.in_dims) => $(cell.out_dims)")
    print(io, ", activation=$(cell.activation)")
    print(io, ", input_activation=$(cell.input_activation)")
    print(io, ", use_bias=$(known(cell.use_bias))")
    cell.exc_recurrence_scale != 0.9  && print(io, ", exc_recurrence_scale=$(cell.exc_recurrence_scale)")
    cell.inh_recurrence_scale != 0.5  && print(io, ", inh_recurrence_scale=$(cell.inh_recurrence_scale)")
    return print(io, ")")
end
