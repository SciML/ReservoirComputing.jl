module RCMLJLinearModelsExt
using ReservoirComputing
using MLJLinearModels

function ReservoirComputing.train(regressor::MLJLinearModels.GeneralizedLinearRegression,
        states::AbstractArray{T}, target::AbstractArray{T};
        kwargs...) where {T <: Number}
    out_size = size(target, 1)
    output_layer = similar(target, size(target, 1), size(states, 1))

    if regressor.fit_intercept
        throw(ArgumentError("fit_intercept=true is not yet supported.
            Please add fit_intercept=false to the MLJ regressor"))
    end

    for i in axes(target, 1)
        output_layer[i, :] = MLJLinearModels.fit(regressor, states',
            target[i, :]; kwargs...)
    end

    return OutputLayer(regressor, output_layer, out_size, target[:, end])
end

end #module
