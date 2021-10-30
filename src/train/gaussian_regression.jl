struct GaussianProcess{M,K,L,O} <: AbstractGaussianProcess
    mean::M
    kernel::K
    lognoise::L
    optimize::Bool
    optimizer::O
end

function GaussianProcess(mean, kernel;
                            lognoise=-2, 
                            optimize=false,
                            optimizer=Optim.LBFGS())
    GaussianProcess(mean, kernel, lognoise, optimize, optimizer)
end

function train(rc::AbstractReservoirComputer, target_data, gp::GaussianProcess)

    states_new = nla(rc.nla_type, rc.states)
    out_size = size(target_data, 1)
    output_matrix = []
    for i=1:out_size
        out_size == 1 ? target = vec(target_data) : target = target_data[i,:]
        push!(output_matrix, GP(states_new, target, gp.mean, gp.kernel, gp.lognoise))
        gp.optimize ? optimize!(output_matrix[i]; method=gp.optimizer) : nothing
    end
    OutputLayer(gp, output_matrix, out_size)
end