
function _train(states, target_data, svr::LIBSVM.AbstractSVR)

    out_size = size(target_data, 1)
    output_matrix = []
    for i=1:out_size
        out_size == 1 ? target = vec(target_data) : target = target_data[i,:]
        push!(output_matrix, LIBSVM.fit!(svr, states', target_data[i,:]))
    end
    OutputLayer(svr, output_matrix, out_size)
end