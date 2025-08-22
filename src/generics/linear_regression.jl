@doc raw"""

    StandardRidge([Type], [reg])

Returns a training method for `train` based on ridge regression.
The equations for ridge regression are as follows:

```math
\mathbf{w} = (\mathbf{X}^\top \mathbf{X} +
\lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
```

# Arguments
 - `Type`: type of the regularization argument. Default is inferred internally,
   there's usually no need to tweak this
 - `reg`: regularization coefficient. Default is set to 0.0 (linear regression).

```
"""
struct StandardRidge
    reg::Number
end

function StandardRidge(::Type{T}, reg) where {T<:Number}
    return StandardRidge(T.(reg))
end

function StandardRidge()
    return StandardRidge(0.0)
end

function train!(rc::ReservoirChain, train_data::AbstractArray,
    target_data::AbstractArray, ps, st::NamedTuple, sr::StandardRidge=StandardRidge(0.0);
    return_states::Bool=false)
    states = collectstates(rc, train_data, ps, st)
    readout = train(sr, states, target_data)
    ps, st = addreadout!(rc, readout, ps, st)

    if return_states
        return (ps, st), states
    else
        return ps, st
    end
end

function train(sr::StandardRidge, states::AbstractArray, target_data::AbstractArray)
    n_states = size(states, 1)
    A = [states'; sqrt(sr.reg) * I(n_states)]
    b = [target_data'; zeros(n_states, size(target_data, 1))]
    F = qr(A)
    Wt = F \ b
    output_layer = Matrix(Wt')
    return output_layer
end

function addreadout!(rc::ReservoirChain, readout_matrix::AbstractArray, ps, st::NamedTuple) #make sure the compile infers
    ro_param = (; weight=readout_matrix)
    new_ps = (;)
    for ((name, layer), param) in zip(pairs(rc.layers), ps)
        if layer isa Readout
            param = merge(param, ro_param)
        end
        new_ps = merge(new_ps, (; name => param))
    end
    return new_ps, st
end

#use a recursion to make it more compiler safe
