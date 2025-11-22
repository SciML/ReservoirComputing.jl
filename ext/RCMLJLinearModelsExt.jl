module RCMLJLinearModelsExt
using ReservoirComputing
using MLJLinearModels

function ReservoirComputing.train(regressor::MLJLinearModels.GeneralizedLinearRegression,
        states::AbstractMatrix{<:Real}, target::AbstractMatrix{<:Real};
        kwargs...)
    @assert size(states, 2)==size(target, 2) "states and target must share the same number of columns."

    if regressor.fit_intercept
        throw(ArgumentError("fit_intercept=true not supported here. \
        Either set fit_intercept=false on the MLJ regressor, or extend addreadout! to write bias."))
    end
    permuted_states = permutedims(states)
    output_matrix = similar(target, size(target, 1), size(states, 1))
    for idx in axes(target, 1)
        yi = vec(target[idx, :])
        coefs = MLJLinearModels.fit(regressor, permuted_states, yi; kwargs...)
        output_matrix[idx, :] = coefs
    end

    return output_matrix
end

end #module
