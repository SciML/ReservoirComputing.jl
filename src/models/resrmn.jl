@doc raw"""
    ResRMN(in_dims, mem_dims, res_dims, out_dims, activation=tanh;
        alpha=1.0, beta=1.0,
        init_memory_reservoir=(rng, dims...) -> simple_cycle(rng, dims...; cycle_weight=1),
        init_memory_input=scaled_rand, init_memory_bias=zeros32,
        init_memory_state=randn32, use_memory_bias=False(),
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_memory=scaled_rand, init_orthogonal=orthogonal,
        init_bias=zeros32, init_state=randn32, use_bias=False(),
        state_modifiers=(), readout_activation=identity)

Residual Reservoir Memory Network (ResRMN) [Ceni2025b](@cite).

`ResRMN` is a modular and hierarchical reservoir computing model that combines
a linear memory reservoir ([`MemoryCell`](@ref)) and a nonlinear residual
reservoir ([`ResRMNCell`](@ref), an extension of [`ResESNCell`](@ref) accepting
a memory input). Both reservoirs are fed the external input `u(t)`; the memory
reservoir output `m(t)` is additionally fed to the nonlinear reservoir.

## Equations

```math
\begin{aligned}
    \mathbf{m}(t) &= \mathbf{W}_{\text{in}}^{m}\, \mathbf{u}(t)
        + \mathbf{C}\, \mathbf{m}(t - 1) + \mathbf{b}^{m}, \\
    \mathbf{h}(t) &= \alpha\, \mathbf{O}\, \mathbf{h}(t - 1)
        + \beta\, \phi\!\left(\mathbf{W}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{W}_{\text{mem}}\, \mathbf{m}(t)
        + \mathbf{W}_r\, \mathbf{h}(t - 1) + \mathbf{b} \right), \\
    \mathbf{z}(t) &= \mathrm{Mods}\!\left(\mathbf{h}(t)\right), \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{\text{out}}\, \mathbf{z}(t)
        + \mathbf{b}_{\text{out}} \right).
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension.
  - `mem_dims`: Memory reservoir dimension.
  - `res_dims`: Nonlinear reservoir dimension.
  - `out_dims`: Output dimension.
  - `activation`: Nonlinear reservoir activation (for [`ResRMNCell`](@ref)).
    Default: `tanh`.

## Keyword arguments

Residual reservoir options (forwarded to [`ResRMNCell`](@ref)):

  - `alpha`, `beta`: Residual skip and nonlinear weights. Default: `1.0`.
  - `init_reservoir`: Initializer for `W_r`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_memory`: Initializer for `W_mem`. Default: [`scaled_rand`](@ref).
  - `init_orthogonal`: Initializer for `O`. Default: `orthogonal`.
  - `init_bias`: Initializer for reservoir bias (used if `use_bias=true`).
    Default: `zeros32`.
  - `init_state`: Initializer used when an external state is not provided.
    Default: `randn32`.
  - `use_bias`: Whether the nonlinear reservoir uses a bias term. Default: `false`.

Memory reservoir options (forwarded to [`MemoryCell`](@ref)):

  - `init_memory_reservoir`: Initializer for `C`. Default: [`simple_cycle`](@ref)
    with unit weight.
  - `init_memory_input`: Initializer for `W_in^m`. Default: [`scaled_rand`](@ref).
  - `init_memory_bias`: Initializer for memory bias (used if `use_memory_bias=true`).
    Default: `zeros32`.
  - `init_memory_state`: Initializer used when an external memory state is not
    provided. Default: `randn32`.
  - `use_memory_bias`: Whether the memory reservoir uses a bias term. Default: `false`.

Composition:

  - `state_modifiers`: A layer or collection of layers applied to the reservoir
    state before the readout. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default: `identity`.

## Inputs

  - `x :: AbstractArray (in_dims, batch)`

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `memory` — parameters of the internal [`MemoryCell`](@ref).
  - `reservoir` — parameters of the internal [`ResRMNCell`](@ref).
  - `states_modifiers` — a `Tuple` with parameters for each modifier layer (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref).

## States

  - `memory` — states for the internal [`MemoryCell`](@ref).
  - `reservoir` — states for the internal [`ResRMNCell`](@ref).
  - `states_modifiers` — a `Tuple` with states for each modifier layer.
  - `readout` — states for [`LinearReadout`](@ref).
"""
@concrete struct ResRMN <:
    AbstractEchoStateNetwork{(:memory, :reservoir, :states_modifiers, :readout)}
    memory
    reservoir
    states_modifiers
    readout
end

function ResRMN(
        in_dims::IntegerType, mem_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        # memory reservoir kwargs
        init_memory_reservoir = _default_memory_reservoir,
        init_memory_input = scaled_rand,
        init_memory_bias = zeros32,
        init_memory_state = randn32,
        use_memory_bias::BoolType = False(),
        # nonlinear reservoir kwargs
        alpha::AbstractFloat = 1.0, beta::AbstractFloat = 1.0,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_memory = scaled_rand, init_orthogonal = orthogonal,
        init_bias = zeros32, init_state = randn32,
        use_bias::BoolType = False(),
        # composition
        state_modifiers = (),
        readout_activation = identity
    )
    memory_cell = MemoryCell(
        in_dims => mem_dims, identity;
        use_bias = use_memory_bias,
        init_bias = init_memory_bias,
        init_reservoir = init_memory_reservoir,
        init_input = init_memory_input,
        init_state = init_memory_state
    )
    res_cell = ResRMNCell(
        (in_dims, mem_dims) => res_dims, activation;
        use_bias = use_bias,
        init_bias = init_bias,
        init_reservoir = init_reservoir,
        init_input = init_input,
        init_memory = init_memory,
        init_orthogonal = init_orthogonal,
        init_state = init_state,
        alpha = alpha,
        beta = beta
    )
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return ResRMN(StatefulLayer(memory_cell), StatefulLayer(res_cell), mods, ro)
end

function initialparameters(rng::AbstractRNG, rm::ResRMN)
    ps_mem = initialparameters(rng, rm.memory)
    ps_res = initialparameters(rng, rm.reservoir)
    ps_mods = map(l -> initialparameters(rng, l), rm.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, rm.readout)
    return (;
        memory = ps_mem, reservoir = ps_res,
        states_modifiers = ps_mods, readout = ps_ro,
    )
end

function initialstates(rng::AbstractRNG, rm::ResRMN)
    st_mem = initialstates(rng, rm.memory)
    st_res = initialstates(rng, rm.reservoir)
    st_mods = map(l -> initialstates(rng, l), rm.states_modifiers) |> Tuple
    st_ro = initialstates(rng, rm.readout)
    return (;
        memory = st_mem, reservoir = st_res,
        states_modifiers = st_mods, readout = st_ro,
    )
end

function _partial_apply(rm::ResRMN, inp, ps, st)
    m_out, st_mem = apply(rm.memory, inp, ps.memory, st.memory)
    h_out, st_res = apply(rm.reservoir, (inp, m_out), ps.reservoir, st.reservoir)
    z, st_mods = _apply_seq(
        rm.states_modifiers, h_out, ps.states_modifiers, st.states_modifiers
    )
    return z, (; memory = st_mem, reservoir = st_res, states_modifiers = st_mods)
end

function (rm::ResRMN)(inp, ps, st)
    out, new_st = _partial_apply(rm, inp, ps, st)
    y, st_ro = apply(rm.readout, out, ps.readout, st.readout)
    return y, merge(new_st, (readout = st_ro,))
end

function collectstates(rm::ResRMN, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    nsteps = size(data, 2)
    cols = eachcol(data)
    @assert !isempty(cols)
    x1 = first(cols)
    current_state, partial_st = _partial_apply(rm, x1, ps, newst)
    state_dims = size(current_state, 1)
    states = similar(data, state_dims, nsteps)
    states[:, 1] .= current_state
    newst = merge(partial_st, (readout = newst.readout,))
    for (idx, inp) in Base.Iterators.drop(Base.enumerate(cols), 1)
        current_state, partial_st = _partial_apply(rm, inp, ps, newst)
        states[:, idx] .= current_state
        newst = merge(partial_st, (readout = newst.readout,))
    end
    return states, newst
end

function collectstates(rm::ResRMN, data::AbstractVector, ps, st::NamedTuple)
    return collectstates(rm, reshape(data, :, 1), ps, st)
end

function resetcarry!(
        rng::AbstractRNG, rm::ResRMN, st; init_carry = nothing
    )
    new_mem = _reset_stateful_carry(rng, rm.memory, st.memory, init_carry)
    new_res = _reset_stateful_carry(rng, rm.reservoir, st.reservoir, init_carry)
    return merge(st, (memory = new_mem, reservoir = new_res))
end

function resetcarry!(
        rng::AbstractRNG, rm::ResRMN, ps, st; init_carry = nothing
    )
    return ps, resetcarry!(rng, rm, st; init_carry = init_carry)
end

@inline function _reset_stateful_carry(
        rng::AbstractRNG, layer, st_layer::NamedTuple, init_carry
    )
    carry = get(st_layer, :carry, nothing)
    sz = if carry === nothing
        _cell_out_dims(layer.cell)
    else
        size(first(carry), 1)
    end
    new_carry = init_carry === nothing ? nothing : (init_carry(rng, sz, 1),)
    return merge(st_layer, (; carry = new_carry))
end

function Base.show(io::IO, rm::ResRMN)
    print(io, "ResRMN(\n")

    print(io, "    memory = ")
    show(io, rm.memory)
    print(io, ",\n")

    print(io, "    reservoir = ")
    show(io, rm.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(rm.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(rm.states_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, rm.readout)
    print(io, "\n)")

    return
end
