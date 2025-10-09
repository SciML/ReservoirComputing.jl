@doc raw"""
    DeepESN(in_dims::Int,
            res_dims::AbstractVector{<:Int},
            out_dims,
            activation=tanh;
            leak_coefficient=1.0,
            init_reservoir=rand_sparse,
            init_input=scaled_rand,
            init_bias=zeros32,
            init_state=randn32,
            use_bias=false,
            state_modifiers=(),
            readout_activation=identity)

Deep Echo State Network (DeepESN): a stack of stateful [`ESNCell`](@ref) layers
(optionally with per-layer state modifiers) followed by a linear readout.

`DeepESN` composes, for `L = length(res_dims)` layers:
  1) a sequence of stateful [`ESNCell`](@ref) with widths `res_dims[ℓ]`,
  2) zero or more per-layer `state_modifiers[ℓ]` applied to the layer's state, and
  3) a final [`LinearReadout`](@ref) from the last layer's features to the output.

## Equations

For input `\mathbf{x}(t) ∈ \mathbb{R}^{in\_dims}`, per-layer reservoir states
`\mathbf{h}^{(\ell)}(t) ∈ \mathbb{R}^{res\_dims[\ell]}` (`\ell = 1..L`), and output
`\mathbf{y}(t) ∈ \mathbb{R}^{out\_dims}`:

```math
\begin{aligned}
    \tilde{\mathbf{h}}^{(1)}(t) &= \phi_1\!\left(
        \mathbf{W}^{(1)}_{in}\,\mathbf{x}(t) + \mathbf{W}^{(1)}_{res}\,\mathbf{h}^{(1)}(t-1)
        + \mathbf{b}^{(1)}\right) \\
    \mathbf{h}^{(1)}(t) &= (1-\alpha_1)\,\mathbf{h}^{(1)}(t-1) + \alpha_1\,\tilde{\mathbf{h}}^{(1)}(t) \\
    \mathbf{u}^{(1)}(t) &= \mathrm{Mods}_1\!\big(\mathbf{h}^{(1)}(t)\big) \\
    \tilde{\mathbf{h}}^{(\ell)}(t) &= \phi_\ell\!\left(
        \mathbf{W}^{(\ell)}_{in}\,\mathbf{u}^{(\ell-1)}(t) +
        \mathbf{W}^{(\ell)}_{res}\,\mathbf{h}^{(\ell)}(t-1) + \mathbf{b}^{(\ell)}\right),
        \quad \ell=2..L \\
    \mathbf{h}^{(\ell)}(t) &= (1-\alpha_\ell)\,\mathbf{h}^{(\ell)}(t-1) + \alpha_\ell\,\tilde{\mathbf{h}}^{(\ell)}(t),
        \quad \ell=2..L \\
    \mathbf{u}^{(\ell)}(t) &= \mathrm{Mods}_\ell\!\big(\mathbf{h}^{(\ell)}(t)\big), \quad \ell=2..L \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{out}\,\mathbf{u}^{(L)}(t) + \mathbf{b}_{out}\right)
\end{aligned}

## Where

- `\mathbf{x}(t) ∈ ℝ^{in_dims × batch}` — input at time `t`.
- `\mathbf{h}^{(\ell)}(t) ∈ ℝ^{res_dims[ℓ] × batch}` — hidden state of layer `ℓ`.
- `\tilde{\mathbf{h}}^{(\ell)}(t)` — candidate state before leaky mixing.
- `\mathbf{u}^{(\ell)}(t)` — features after applying the `ℓ`-th `state_modifiers` (identity if none).
- `\mathbf{y}(t) ∈ ℝ^{out_dims × batch}` — network output.

- `\mathbf{W}^{(\ell)}_{in} ∈ ℝ^{res_dims[ℓ] × in\_size[ℓ]}` — input matrix at layer `ℓ`
  (`in_size[1]=in_dims`, `in_size[ℓ]=res_dims[ℓ-1]` for `ℓ>1`).
- `\mathbf{W}^{(\ell)}_{res} ∈ ℝ^{res_dims[ℓ] × res_dims[ℓ]}` — reservoir matrix at layer `ℓ`.
- `\mathbf{b}^{(\ell)} ∈ ℝ^{res_dims[ℓ] × 1}` — reservoir bias (broadcast over batch), present iff `use_bias[ℓ]=true`.
- `\mathbf{W}_{out} ∈ ℝ^{out_dims × res_dims[L]}` — readout matrix.
- `\mathbf{b}_{out} ∈ ℝ^{out_dims × 1}` — readout bias (if used by the readout).

- `\phi_\ell` — activation of layer `ℓ` (`activation[ℓ]`, default `tanh`).
- `\alpha_\ell ∈ (0,1]` — leak coefficient of layer `ℓ` (`leak_coefficient[ℓ]`).
- `\mathrm{Mods}_\ell(·)` — composition of modifiers for layer `ℓ` (may be empty).
- `\rho` — readout activation (`readout_activation`, default `identity`).

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
    Scalar or length-`L`. Default: [`zeros32`](@extref).
  - `init_state`: Initializer(s) used when an external state is not provided.
    Scalar or length-`L`. Default: [`randn32`](@extref).
  - `use_bias`: Whether each reservoir uses a bias term. Boolean scalar or length-`L`. Default: `false`.

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
      - `input_matrix :: (res_dims[ℓ] × in_size[ℓ])` — `W_in^{(ℓ)}`
      - `reservoir_matrix :: (res_dims[ℓ] × res_dims[ℓ])` — `W_res^{(ℓ)}`
      - `bias :: (res_dims[ℓ],)` — present only if `use_bias[ℓ]=true`
  - `states_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier parameters (empty tuples if none).
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × res_dims[L])` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

> Exact field names for modifiers/readout follow their respective layer definitions.

## States

  - `cells :: NTuple{L,NamedTuple}` — states for each [`ESNCell`](@ref).
  - `states_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier states.
  - `readout` — states for [`LinearReadout`](@ref).

"""
@concrete struct DeepESN <: AbstractEchoStateNetwork{(:cells, :states_modifiers, :readout)}
    cells
    states_modifiers
    readout
end

function DeepESN(in_dims::IntegerType,
    res_dims::AbstractVector{<:IntegerType},
    out_dims::IntegerType,
    activation=tanh;
    leak_coefficient=1.0,
    init_reservoir=rand_sparse,
    init_input=scaled_rand,
    init_bias=zeros32,
    init_state=randn32,
    use_bias=false,
    state_modifiers=(),
    readout_activation=identity)

    n_layers = length(res_dims)
    acts = _asvec(activation, n_layers)
    leaks = _asvec(leak_coefficient, n_layers)
    ires = _asvec(init_reservoir, n_layers)
    iinp = _asvec(init_input, n_layers)
    ibias = _asvec(init_bias, n_layers)
    ist = _asvec(init_state, n_layers)
    ub = _asvec(use_bias, n_layers)
    mods0 = _asvec(state_modifiers, n_layers)

    cells = Vector{Any}(undef, n_layers)
    states_modifiers = Vector{Any}(undef, n_layers)

    prev = in_dims
    for idx in firstindex(res_dims):lastindex(res_dims)
        cell = ESNCell(prev => res_dims[idx], acts[idx];
            use_bias=static(ub[idx]),
            init_bias=ibias[idx],
            init_reservoir=ires[idx],
            init_input=iinp[idx],
            init_state=ist[idx],
            leak_coefficient=leaks[idx])
        cells[idx] = StatefulLayer(cell)
        states_modifiers[idx] = mods0[idx] === nothing ? nothing : _wrap_layer(mods0[idx])
        prev = res_dims[idx]
    end
    mods_per_layer = map(_coerce_layer_mods, states_modifiers) |> Tuple
    ro = LinearReadout(prev => out_dims, readout_activation)
    return DeepESN(Tuple(cells), mods_per_layer, ro)
end

DeepESN(in_dims::Int, res_dim::Int, out_dims::Int; depth::Int=2, kwargs...) =
    DeepESN(in_dims, fill(res_dim, depth), out_dims; kwargs...)

function initialparameters(rng::AbstractRNG, desn::DeepESN)
    ps_cells = map(l -> initialparameters(rng, l), desn.cells) |> Tuple
    mods = desn.states_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
           desn.states_modifiers
    ps_mods = map(layer_mods ->
            (layer_mods === nothing ? () :
             map(l -> initialparameters(rng, l), layer_mods) |> Tuple),
        mods) |> Tuple

    ps_ro = initialparameters(rng, desn.readout)
    return (cells=ps_cells, states_modifiers=ps_mods, readout=ps_ro)
end

function initialstates(rng::AbstractRNG, desn::DeepESN)
    st_cells = map(l -> initialstates(rng, l), desn.cells) |> Tuple

    mods = desn.states_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
           desn.states_modifiers

    st_mods = map(layer_mods ->
            (layer_mods === nothing ? () :
             map(l -> initialstates(rng, l), layer_mods) |> Tuple),
        mods) |> Tuple

    st_ro = initialstates(rng, desn.readout)
    return (cells=st_cells, states_modifiers=st_mods, readout=st_ro)
end

function _partial_apply(desn::DeepESN, inp, ps, st)
    inp_t = inp
    n_layers = length(desn.cells)
    new_cell_st = Vector{Any}(undef, n_layers)
    new_mods_st = Vector{Any}(undef, n_layers)
    for idx in firstindex(desn.cells):lastindex(desn.cells)
        inp_t, st_cell_i = apply(desn.cells[idx], inp_t, ps.cells[idx], st.cells[idx])
        new_cell_st[idx] = st_cell_i
        inp_t, st_mods_i = _apply_seq(desn.states_modifiers[idx], inp_t,
            ps.states_modifiers[idx], st.states_modifiers[idx])
        new_mods_st[idx] = st_mods_i
    end

    return inp_t, (;
        cells=tuple(new_cell_st...),
        states_modifiers=tuple(new_mods_st...),
    )
end

function (desn::DeepESN)(inp, ps, st)
    out, new_st = _partial_apply(desn, inp, ps, st)
    inp_t, st_ro = apply(desn.readout, out, ps.readout, st.readout)
    return inp_t, merge(new_st, (readout=st_ro,))
end

function resetcarry!(rng::AbstractRNG, desn::DeepESN, st; init_carry=nothing)
    n_layers = length(desn.cells)

    @inline function _layer_outdim(idx)
        st_i = st.cells[idx]
        if st_i.carry === nothing
            return desn.cells[idx].cell.out_dims
        else
            return size(first(st_i.carry), 1)
        end
    end

    @inline function _init_for(idx)
        if init_carry === nothing
            return nothing
        elseif init_carry isa Function
            sz = _layer_outdim(idx)
            return (_asvec(init_carry(rng, sz)),)
        elseif init_carry isa Tuple || init_carry isa AbstractVector
            f = init_carry[idx]
            sz = _layer_outdim(idx)
            return f === nothing ? nothing : (_asvec(f(rng, sz)),)
        else
            throw(ArgumentError("init_carry must be nothing, a Function, or a Tuple/Vector of Functions"))
        end
    end

    new_cells = ntuple(idx -> begin
            st_i = st.cells[idx]
            new_carry = _init_for(idx)
            merge(st_i, (; carry=new_carry))
        end, n_layers)

    return (;
        cells=new_cells,
        states_modifiers=st.states_modifiers,
        readout=st.readout,
    )
end

function collectstates(desn::DeepESN, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    n_layers = length(desn.cells)
    for inp in eachcol(data)
        inp_t = inp
        cell_st_parts = Vector{Any}(undef, n_layers)
        mods_st_parts = Vector{Any}(undef, n_layers)
        for idx in firstindex(desn.cells):lastindex(desn.cells)
            inp_t, st_cell_i = apply(desn.cells[idx], inp_t, ps.cells[idx], newst.cells[idx])
            cell_st_parts[idx] = st_cell_i
            inp_t, st_mods_i = _apply_seq(
                desn.states_modifiers[idx], inp_t,
                ps.states_modifiers[idx], newst.states_modifiers[idx]
            )
            mods_st_parts[idx] = st_mods_i
        end
        push!(collected, copy(inp_t))
        newst = (;
            cells=tuple(cell_st_parts...),
            states_modifiers=tuple(mods_st_parts...),
            readout=newst.readout,
        )
    end
    @assert !isempty(collected)
    states = eltype(data).(reduce(hcat, collected))

    return states, newst
end

collectstates(m::DeepESN, data::AbstractVector, ps, st::NamedTuple) =
    collectstates(m, reshape(data, :, 1), ps, st)
