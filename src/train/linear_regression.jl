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

function StandardRidge(::Type{T}, reg) where {T <: Number}
    return StandardRidge(T.(reg))
end

function StandardRidge()
    return StandardRidge(0.0)
end

function train(sr::StandardRidge, states::AbstractArray, target_data::AbstractArray)
    #A = states * states' + sr.reg * I
    #b = states * target_data
    #output_layer = (A \ b)'

    if size(states, 2) != size(target_data, 2)
        throw(DimensionMismatch("\n" *
                                "\n" *
                                "  - Number of columns in `states`: $(size(states, 2))\n" *
                                "  - Number of columns in `target_data`: $(size(target_data, 2))\n" *
                                "The dimensions of `states` and `target_data` must align for training." *
                                "\n"
        ))
    end

    output_layer = Matrix(((states * states' + sr.reg * I) \
                           (states * target_data'))')
    return OutputLayer(sr, output_layer, size(target_data, 1), target_data[:, end])
end
