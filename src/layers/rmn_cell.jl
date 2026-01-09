"""
    RMNCell(nonlinear_reservoir, linear_reservoir)
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
    return (nonlinear_reservoir = nlr_st, linear_reservoir = lr_st)
end

function (rmn::RMNCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, rmn, inp)
    memory_state = init_hidden_state(rng, rmn, inp)
    return rmn((inp, (hidden_state, memory_state)), ps, merge(st, (; rng)))
end

function (rmn::RMNCell)((inp, (hidden_state, memory_state))::InputType, ps, st::NamedTuple)
    (mstate_new, _), st_lin = rmn.linear_reservoir((inp, (memory_state,)), ps.linear_reservoir, st.linear_reservoir)
    (hstate_new, _), st_nonlin = rmn.nonlinear_reservoir((inp, (hidden_state, mstate_new)), ps.nonlinear_reservoir, st.nonlinear_reservoir)
    return (hstate_new, (hstate_new, mstate_new)), st
end

function Base.show(io::IO, rmn::RMNCell)
    ### TODO
end
