struct StandardRidge
    reg::Number
end

function StandardRidge(::Type{T}, reg) where {T <: Number}
    return StandardRidge(T.(reg))
end

function StandardRidge()
    return StandardRidge(0.0)
end

function train(sr::StandardRidge,
        states,
        target_data)
    #A = states * states' + sr.reg * I
    #b = states * target_data
    #output_layer = (A \ b)'
    output_layer = Matrix(((states * states' + sr.reg * I) \
                           (states * target_data'))')
    return OutputLayer(sr, output_layer, size(target_data, 1), target_data[:, end])
end
