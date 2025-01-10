module RCLIBSVMExt
using ReservoirComputing
using LIBSVM

function ReservoirComputing.train(svr::LIBSVM.AbstractSVR,
        states::AbstractArray, target::AbstractArray)
    out_size = size(target, 1)
    output_matrix = []

    if out_size == 1
        output_matrix = LIBSVM.fit!(svr, states', vec(target))
    else
        for i in 1:out_size
            push!(output_matrix, LIBSVM.fit!(svr, states', target[i, :]))
        end
    end

    return OutputLayer(svr, output_matrix, out_size, target[:, end])
end

function ReservoirComputing.get_prediction(training_method::LIBSVM.AbstractSVR,
        output_layer::AbstractArray, x::AbstractArray)
    out = zeros(output_layer.out_size)

    for i in 1:(output_layer.out_size)
        x_new = reshape(x, 1, length(x))
        out[i] = LIBSVM.predict(output_layer.output_matrix[i], x_new)[1]
    end

    return out
end

end #module
