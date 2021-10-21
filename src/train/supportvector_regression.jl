
function train(rc::AbstractReservoirComputer, target_data, svr::LIBSVM.AbstractSVR)

    states_new = nla(rc.nla_type, rc.states)
    out_size = size(target_data, 1)
    output_matrix = []
    for i=1:out_size
        out_size == 1 ? target = vec(target_data) : target = target_data[i,:]
        push!(output_matrix, LIBSVM.fit!(svr, states_new', target_data[i,:]))
    end
    OutputLayer(svr, output_matrix, out_size)
end