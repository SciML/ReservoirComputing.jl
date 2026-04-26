@doc raw"""
    RMN(in_dims, mem_dims,res_dims, out_dims, activation=tanh;
        leak_coefficient=1.0,
        init_memory_reservoir=simple_cycle(; cycle_weight=1),
        init_memory_input=scaled_rand, init_memory_bias=zeros32,
        init_memory_state=randn32, use_memory_bias=False(),
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_memory=scaled_rand, init_orthogonal=orthogonal,
        init_bias=zeros32, init_state=randn32, use_bias=False(),
        state_modifiers=(), readout_activation=identity)
"""
@concrete struct RMNESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function RMNESN(
        in_dims::IntegerType, mem_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        init_memory_reservoir=simple_cycle(; cycle_weight=1),
        init_memory_input=scaled_rand, init_memory_bias=zeros32,
        init_memory_state=randn32, use_memory_bias=False(),
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_memory=scaled_rand, init_orthogonal=orthogonal,
        init_bias=zeros32, init_state=randn32, use_bias=False(),
        state_modifiers=(), readout_activation=identity
    )
    linear_reservoir = ESNCell(in_dims => mem_dims, identity;
        use_bias = use_memory_bias,
        init_bias = init_memory_bias,
        init_reservoir = init_memory_reservoir,
        init_input = init_memory_input,
        init_state = init_memory_state
    )
    nonlinear_reservoir = MemoryESNCell((in_dims, mem_dims) => res_dims;
        use_bias = use_bias,
        init_bias = init_bias,
        init_reservoir = init_reservoir,
        init_input = init_input,
        init_memory = init_memory,
        init_state = init_state,
    )
    rmncell = RMNCell(nonlinear_reservoir, linear_reservoir)
    cell = StatefulLayer(rmncell)
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return RMNESN(cell, mods, ro)
end
