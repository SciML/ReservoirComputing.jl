@doc raw"""
    DeepESN(in_dims, res_dims, out_dims,
            activation=tanh; depth=2, leak_coefficient=1.0, init_reservoir=rand_sparse,
            init_input=scaled_rand, init_bias=zeros32, init_state=randn32,
            use_bias=false, state_modifiers=(), readout_activation=identity)

Deep Echo State Network [Gallicchio2017](@cite).

`DeepESN` composes, for `L = length(res_dims)` layers:
  1) a sequence of stateful [`ESNCell`](@ref) with widths `res_dims[ℓ]`,
  2) zero or more per-layer `state_modifiers[ℓ]` applied to the layer's state, and
  3) a final [`LinearReadout`](@ref) from the last layer's features to the output.

## Equations

```math
\begin{aligned}
    \mathbf{x}^{(1)}(t) &= (1-\alpha_1)\, \mathbf{x}^{(1)}(t-1)
        + \alpha_1\, \phi_1\!\left(\mathbf{W}^{(1)}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{W}^{(1)}_r\, \mathbf{x}^{(1)}(t-1) + \mathbf{b}^{(1)} \right), \\
    \mathbf{u}^{(1)}(t) &= \mathrm{Mods}_1\!\left(\mathbf{x}^{(1)}(t)\right), \\
    \mathbf{x}^{(\ell)}(t) &= (1-\alpha_\ell)\, \mathbf{x}^{(\ell)}(t-1)
        + \alpha_\ell\, \phi_\ell\!\left(\mathbf{W}^{(\ell)}_{\text{in}}\,
        \mathbf{u}^{(\ell-1)}(t) + \mathbf{W}^{(\ell)}_r\, \mathbf{x}^{(\ell)}(t-1)
        + \mathbf{b}^{(\ell)} \right), \quad \ell = 2,\dots,L, \\
    \mathbf{u}^{(\ell)}(t) &= \mathrm{Mods}_\ell\!\left(\mathbf{x}^{(\ell)}(t)\right),
        \quad \ell = 2,\dots,L, \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{\text{out}}\, \mathbf{u}^{(L)}(t)
        + \mathbf{b}_{\text{out}} \right).
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Vector of reservoir (hidden) dimensions per layer; its length sets the depth `L`.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation(s). Either a single function (broadcast to all layers)
    or a vector/tuple of length `L`. Default: `tanh`.

## Keyword arguments

Per-layer reservoir options (passed to each [`ESNCell`](@ref)):

  - `leak_coefficient`: Leak rate(s) `α_ℓ ∈ (0,1]`. Scalar or length-`L` collection. Default: `1.0`.
  - `init_reservoir`: Initializer(s) for `W_res^{(ℓ)}`. Scalar or length-`L`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer(s) for `W_in^{(ℓ)}`. Scalar or length-`L`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initializer(s) for reservoir bias (used iff `use_bias[ℓ]=true`).
    Scalar or length-`L`. Default: `zeros32`.
  - `init_state`: Initializer(s) used when an external state is not provided.
    Scalar or length-`L`. Default: `randn32`.
  - `use_bias`: Whether each reservoir uses a bias term. Boolean scalar or length-`L`. Default: `false`.

Depth:

  - `depth`: Number of reservoir layers. Only used when `res_dims` is given as a
    single integer (the depth is then `depth` layers of that width); it is ignored
    when `res_dims` is a vector, whose length already sets the depth `L`. Default: `2`.

Composition:

  - `state_modifiers`: Per-layer modifier(s) applied to each layer’s state before it
    feeds into the next layer (and the readout for the last layer). Accepts `nothing`,
    a single layer, a vector/tuple of length `L`, or per-layer collections. Defaults to no modifiers.
  - `readout_activation`: Activation for the final linear readout. Default: `identity`.

## Inputs

  - `x :: AbstractArray (in_dims, batch)`

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (NamedTuple) containing states for all cells, modifiers, and readout.

## Parameters

  - `cells :: NTuple{L,NamedTuple}` — parameters for each [`ESNCell`](@ref), including:
    + `input_matrix :: (res_dims[ℓ] × in_size[ℓ])` — `W_in^{(ℓ)}`
    + `reservoir_matrix :: (res_dims[ℓ] × res_dims[ℓ])` — `W_res^{(ℓ)}`
    + `bias :: (res_dims[ℓ],)` — present only if `use_bias[ℓ]=true`
  - `state_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier parameters (empty tuples if none).
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
    + `weight :: (out_dims × res_dims[L])` — `W_out`
    + `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

> Exact field names for modifiers/readout follow their respective layer definitions.

## States

  - `cells :: NTuple{L,NamedTuple}` — states for each [`ESNCell`](@ref).
  - `state_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier states.
  - `readout` — states for [`LinearReadout`](@ref).

"""
@concrete struct DeepESN <: AbstractEchoStateNetwork{(:cells, :state_modifiers, :readout)}
    cells
    state_modifiers
    readout
end

function DeepESN(
        in_dims::IntegerType,
        res_dims::AbstractVector{<:IntegerType},
        out_dims::IntegerType,
        activation = tanh;
        leak_coefficient = 1.0,
        init_reservoir = rand_sparse,
        init_input = scaled_rand,
        init_bias = zeros32,
        init_state = randn32,
        use_bias = false,
        state_modifiers = (),
        readout_activation = identity
    )
    n_layers = length(res_dims)
    acts = __asvec(activation, n_layers)
    leaks = __asvec(leak_coefficient, n_layers)
    ires = __asvec(init_reservoir, n_layers)
    iinp = __asvec(init_input, n_layers)
    ibias = __asvec(init_bias, n_layers)
    istate = __asvec(init_state, n_layers)
    ub = __asvec(use_bias, n_layers)
    mods0 = __asvec(state_modifiers, n_layers)

    cells = ntuple(n_layers) do idx
        input_dims = idx == firstindex(res_dims) ? in_dims : res_dims[idx - 1]
        cell = ESNCell(
            input_dims => res_dims[idx], acts[idx];
            use_bias = static(ub[idx]),
            init_bias = ibias[idx],
            init_reservoir = ires[idx],
            init_input = iinp[idx],
            init_state = istate[idx],
            leak_coefficient = leaks[idx]
        )
        StatefulLayer(cell)
    end
    state_modifiers = ntuple(n_layers) do idx
        mods = mods0[idx]
        mods === nothing ? nothing : __wrap_layer(mods)
    end
    mods_per_layer = map(__coerce_layer_mods, state_modifiers) |> Tuple
    ro = LinearReadout(last(res_dims) => out_dims, readout_activation)
    return DeepESN(cells, mods_per_layer, ro)
end

function DeepESN(
        in_dims::Int, res_dim::Int, out_dims::Int,
        activation = tanh; depth::Int = 2, kwargs...
    )
    return DeepESN(in_dims, fill(res_dim, depth), out_dims, activation; kwargs...)
end

function initialparameters(rng::AbstractRNG, desn::DeepESN)
    ps_cells = map(l -> initialparameters(rng, l), desn.cells) |> Tuple
    mods = desn.state_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
        desn.state_modifiers
    ps_mods = map(
        layer_mods -> (
            layer_mods === nothing ? () :
                map(l -> initialparameters(rng, l), layer_mods) |> Tuple
        ),
        mods
    ) |> Tuple

    ps_ro = initialparameters(rng, desn.readout)
    return (cells = ps_cells, state_modifiers = ps_mods, readout = ps_ro)
end

function initialstates(rng::AbstractRNG, desn::DeepESN)
    st_cells = map(l -> initialstates(rng, l), desn.cells) |> Tuple

    mods = desn.state_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
        desn.state_modifiers

    st_mods = map(
        layer_mods -> (
            layer_mods === nothing ? () :
                map(l -> initialstates(rng, l), layer_mods) |> Tuple
        ),
        mods
    ) |> Tuple

    st_ro = initialstates(rng, desn.readout)
    return (cells = st_cells, state_modifiers = st_mods, readout = st_ro)
end

function __partial_apply(desn::DeepESN, inp, ps, st)
    out, cells_st, modifiers_st = __apply_deep_layers(
        desn.cells, desn.state_modifiers, inp,
        ps.cells, ps.state_modifiers, st.cells, st.state_modifiers
    )
    return out, (; cells = cells_st, state_modifiers = modifiers_st)
end

function (desn::DeepESN)(inp, ps, st)
    out, new_st = __partial_apply(desn, inp, ps, st)
    inp_t, st_ro = apply(desn.readout, out, ps.readout, st.readout)
    return inp_t, merge(new_st, (readout = st_ro,))
end

function resetcarry!(rng::AbstractRNG, desn::DeepESN, st; init_carry = nothing)
    n_layers = length(desn.cells)

    @inline function __layer_outdim(idx)
        st_i = st.cells[idx]
        if st_i.carry === nothing
            return desn.cells[idx].cell.out_dims
        else
            return size(first(st_i.carry), 1)
        end
    end

    @inline function __init_for(idx)
        if init_carry === nothing
            return nothing
        elseif init_carry isa Function
            sz = __layer_outdim(idx)
            return (__asvec(init_carry(rng, sz)),)
        elseif init_carry isa Tuple || init_carry isa AbstractVector
            f = init_carry[idx]
            sz = __layer_outdim(idx)
            return f === nothing ? nothing : (__asvec(f(rng, sz)),)
        else
            throw(ArgumentError("init_carry must be nothing, a Function, or a Tuple/Vector of Functions"))
        end
    end

    new_cells = ntuple(
        idx -> begin
            st_i = st.cells[idx]
            new_carry = __init_for(idx)
            merge(st_i, (; carry = new_carry))
        end, n_layers
    )

    return (;
        cells = new_cells,
        state_modifiers = st.state_modifiers,
        readout = st.readout,
    )
end

function collectstates(desn::DeepESN, data::AbstractMatrix, ps, st::NamedTuple)
    __require_nonempty_data(data, "collectstates")
    cols = eachcol(data)
    first_state, partial_st = __partial_apply(desn, first(cols), ps, st)
    states = similar(first_state, eltype(first_state), length(first_state), size(data, 2))
    states[:, 1] .= first_state
    newst = merge(partial_st, (; readout = st.readout))
    for (idx, inp) in Base.Iterators.drop(Base.enumerate(cols), 1)
        current_state, partial_st = __partial_apply(desn, inp, ps, newst)
        states[:, idx] .= current_state
        newst = merge(partial_st, (; readout = newst.readout))
    end

    return states, newst
end

function collectstates(m::DeepESN, data::AbstractVector, ps, st::NamedTuple)
    return collectstates(m, reshape(data, :, 1), ps, st)
end
