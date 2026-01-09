@doc raw"""
    RMN(in_dims, res_dims, out_dims, activation=tanh;
        leak_coefficient=1.0, init_reservoir=rand_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=randn32, use_bias=false,
        state_modifiers=(), readout_activation=identity)

"""
@concrete struct RMN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function RMN(
        in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        readout_activation = identity,
        state_modifiers = (),
        kwargs...
    )
    linear_reservoir = ESNCell(in_dims=>res_dims, identity; )
    nonlinear_reservoir = MemoryESNCell(in_dims=>res_dims; kwargs...)
    rmncell = RMNCell(nonlinear_reservoir, linear_reservoir)
    cell = StatefulLayer(rmncell)
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return RMN(cell, mods, ro)
end
