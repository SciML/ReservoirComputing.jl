@doc raw"""
    NGRC(in_dims, out_dims; num_delays=2, stride=1,
         features=(), include_input=true, init_delay=zeros32,
         readout_activation=identity, state_modifiers=(),
         ro_dims=nothing)

Next Generation Reservoir Computing (NGRC) / NVAR-style model [Gauthier2021](@cite):
a tapped-delay embedding of the input, followed by user-defined nonlinear feature
maps and a linear readout. This is a "reservoir-free" architecture where all dynamics
come from explicit input delays rather than a recurrent state.

`NGRC` composes:
  1) a [`DelayLayer`](@ref) applied directly to the input, producing a vector
     containing the current input and a fixed number of past inputs,
  2) a [`NonlinearFeaturesLayer`](@ref) that applies user-provided functions to
     this delayed vector and concatenates the results, and
  3) a [`LinearReadout`](@ref) mapping the resulting feature vector to outputs.

Internally, `NGRC` is represented as a [`ReservoirComputer`](@ref) with:
  - `reservoir` = the [`DelayLayer`](@ref),
  - `states_modifiers` = the [`NonlinearFeaturesLayer`](@ref) plus any extra
    `state_modifiers`,
  - `readout` = the [`LinearReadout`](@ref).

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Output dimension.

## Keyword arguments

  - `num_delays`: Number of past input vectors to include. The internal
    [`DelayLayer`](@ref) outputs a vector of length
    `(num_delays + 1) * in_dims` (current input plus `num_delays` past inputs).
    Default: `2`.
  - `stride`: Delay stride in layer calls. The delay buffer is updated only when
    the internal clock is a multiple of `stride`. Default: `1`.
  - `init_delay`: Initializer (or tuple of initializers) for the delay history,
    passed to [`DelayLayer`](@ref). Each initializer function is called as
    `init(rng, in_dims, 1)` to fill one delay column. Default: `zeros32`.
  - `features`: A function or tuple of functions `(f₁, f₂, ...)` used by
    [`NonlinearFeaturesLayer`](@ref). Each `f` is called as `f(x)` where `x` is
    the delayed input vector. By default it is assumed that each `f` returns a
    vector of the same length as `x` when `ro_dims` is not provided.
    Default: empty `()`.
  - `include_input`: Whether to include the raw delayed input vector itself as
    the first block of the feature vector (passed to
    [`NonlinearFeaturesLayer`](@ref)). Default: `true`.
  - `state_modifiers`: Extra layers applied after the `NonlinearFeaturesLayer`
    and before the readout. Accepts a single layer, an `AbstractVector`, or a
    `Tuple`. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default: `identity`.
  - `ro_dims`: Input dimension of the readout. If `nothing` (default), it is
    *estimated* under the assumption that each feature function returns a
    vector with the same length as the delayed input. In that case,
    `ro_dims ≈ (num_delays + 1) * in_dims * n_blocks`, where `n_blocks` is the
    number of concatenated vectors (original delayed input if
    `include_input=true` plus one block per feature function).
    If your feature functions change the length (e.g. constant features,
    higher-order polynomial expansions with cross terms), you should pass
    `ro_dims` explicitly.

## Inputs

  - `x :: AbstractArray (in_dims, batch)` or `(in_dims,)`

## Returns

  - Output `y :: (out_dims, batch)` (or `(out_dims,)` for vector input).
  - Updated layer state (NamedTuple).
"""
@concrete struct NGRC <:
                 AbstractReservoirComputer{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function NGRC(in_dims::IntegerType, out_dims::IntegerType; num_delays::IntegerType = 2,
        stride::IntegerType = 1, features = (), include_input::BoolType = True(), init_delay = zeros32,
        readout_activation = identity, state_modifiers = (), ro_dims = nothing)
    reservoir = DelayLayer(in_dims; num_delays = Int(num_delays), stride = Int(stride), init_delay = init_delay)
    feats_tuple = features isa Tuple ? features : (features,)
    nfl = NonlinearFeaturesLayer(feats_tuple...; include_input = include_input)
    mods_tuple_raw = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                     (nfl, state_modifiers...) : (nfl, state_modifiers)
    mods = _wrap_layers(mods_tuple_raw)
    if ro_dims === nothing
        n_taps = in_dims * (num_delays + 1)
        inc = ReservoirComputing.known(include_input)
        n_blocks = (inc === true ? 1 : 0) + length(feats_tuple)
        ro_dims = n_taps * n_blocks
        @warn """
            NGRC: inferring readout input dimension assuming each feature f in `features`
            returns a vector of the same length as the delayed input.

            Delayed input length: n_taps = $(n_taps)
            Blocks (input + features): $(n_blocks)
            => ro_dims = n_taps * n_blocks = $(ro_dims)

            If your feature functions change the length (e.g. constant features,
            quadratic monomials with cross terms), please pass `ro_dims` explicitly.

            Please note that, if dimensions are not correct, training will change them and
            no error will occur.
        """
    end
    readout = LinearReadout(ro_dims => out_dims, readout_activation)

    return NGRC(reservoir, mods, readout)
end

function resetcarry!(
        rng::AbstractRNG, rc::NGRC, st; init_carry = nothing)
    carry = get(st.reservoir, :carry, nothing)
    @warn("""
        Next generation reservoir computing has no internal state to reset.
        Returning untouched model states.
        """)
    return st
end

@doc raw"""
    polynomial_monomials(input_vector;
        degrees = 1:2)

Generate all unordered polynomial monomials of the entries in `input_vector`
for the given set of degrees.

For each `d` in `degrees`, this function produces all degree-`d` monomials
of the form

- degree 1: `x₁, x₂, …`
- degree 2: `x₁², x₁x₂, x₁x₃, x₂², …`
- degree 3: `x₁³, x₁²x₂, x₁x₂x₃, x₂³, …`

where combinations are taken with repetition and in non-decreasing index
order. This means that, for example, `x₁x₂` and `x₂x₁` are represented only
once.

The returned vector is a flat list of all such products, in a deterministic
order determined by the recursive enumeration.

## Arguments

- `input_vector`
  Input vector whose entries define the variables used to build monomials.

## Keyword arguments

- `degrees`: An iterable of positive integers specifying which monomial degrees
  to generate. Each degree less than `1` is skipped. Default: `1:2`.

## Returns

- `output_monomials` a vector of the same type as `input_vector`
  containing all generated monomials, concatenated across the requested
  degrees, in a deterministic order.
"""
function polynomial_monomials(input_vector::AbstractVector;
        degrees = 1:2)
    element_type = eltype(input_vector)
    output_monomials = element_type[]
    num_variables = length(input_vector)
    for degree in degrees
        degree < 1 && continue
        index_buffer = Vector{Int}(undef, degree)
        _polynomial_monomials_recursive!(output_monomials, input_vector,
            index_buffer, 1, 1, num_variables
        )
    end

    return output_monomials
end

function _polynomial_monomials_recursive!(output_monomials, input_vector,
        index_buffer, position::Int, start_index::Int, num_variables::Int)
    if position > length(index_buffer)
        element_type = eltype(input_vector)
        product_value = one(element_type)
        @inbounds for variable_index in index_buffer
            product_value *= input_vector[variable_index]
        end
        push!(output_monomials, product_value)
    else
        @inbounds for variable_index in start_index:num_variables
            index_buffer[position] = variable_index
            _polynomial_monomials_recursive!(output_monomials, input_vector,
                index_buffer, position + 1, variable_index, num_variables)
        end
    end
end
