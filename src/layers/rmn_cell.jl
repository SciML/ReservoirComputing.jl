"""
    RMNCell(nonlinear_reservoir, linear_reservoir)

Reservoir Memory Network recurrent cell [Gallicchio2024b](@cite)
composed of a nonlinear reservoir and a linear memory reservoir.

`RMNCell` maintains two recurrent states:

  - a nonlinear hidden state, updated by `nonlinear_reservoir`
  - a linear memory state, updated by `linear_reservoir`

At each recurrent step, the linear reservoir first updates the memory state from
the current input. The resulting memory state is then supplied to the nonlinear
reservoir, which updates the hidden state.

The cell returns the nonlinear hidden state as its output and stores both the
hidden and memory states for the next recurrent step.

# Arguments

  - `nonlinear_reservoir`: reservoir cell that updates the nonlinear hidden state.
    It must accept inputs of the form `(inp, (hidden_state, memory_state))`.
  - `linear_reservoir`: reservoir cell that updates the linear memory state.
    It must accept inputs of the form `(inp, (memory_state,))`.

# Returns

An `RMNCell` suitable for use inside [`StatefulLayer`](@ref).

# See also

[`MemoryESNCell`](@ref), [`RMNESN`](@ref)
"""
@concrete struct RMNCell <: AbstractReservoirRecurrentCell
    nonlinear_reservoir
    linear_reservoir
end

function initialparameters(rng::AbstractRNG, rmn::RMNCell)
    nlr_ps = initialparameters(rng, rmn.nonlinear_reservoir)
    lr_ps = initialparameters(rng, rmn.linear_reservoir)
    return (nonlinear_reservoir = nlr_ps, linear_reservoir = lr_ps)
end

function initialstates(rng::AbstractRNG, rmn::RMNCell)
    nlr_st = initialstates(rng, rmn.nonlinear_reservoir)
    lr_st = initialstates(rng, rmn.linear_reservoir)
    return (nonlinear_reservoir = nlr_st, linear_reservoir = lr_st, rng = rng)
end

function (rmn::RMNCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, rmn.nonlinear_reservoir, inp)
    memory_state = init_hidden_state(rng, rmn.linear_reservoir, inp)
    return rmn((inp, (hidden_state, memory_state)), ps, merge(st, (; rng)))
end

function (rmn::RMNCell)((inp, (hidden_state, memory_state))::Tuple{AbstractArray, Tuple{AbstractArray, AbstractArray}},
        ps, st::NamedTuple)
    (mstate_new, _), st_lin = rmn.linear_reservoir((inp, (memory_state,)), ps.linear_reservoir, st.linear_reservoir)
    (hstate_new, _), st_nonlin = rmn.nonlinear_reservoir((inp, (hidden_state, mstate_new)), ps.nonlinear_reservoir, st.nonlinear_reservoir)
    return (hstate_new, (hstate_new, mstate_new)), st
end

function Base.show(io::IO, rmn::RMNCell)
    print(io, "RMNCell(")
    print(io, "\n  nonlinear_reservoir = ")
    show(io, rmn.nonlinear_reservoir)
    print(io, ",\n  linear_reservoir = ")
    show(io, rmn.linear_reservoir)
    return print(io, "\n)")
end
