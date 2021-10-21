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
    if out_size == 1
        output_matrix = GP(states_new, vec(target_data), gp.mean, gp.kernel, gp.lognoise)
        gp.optimize ? optimize!(output_matrix; method=gp.optimizer) : nothing
    else
        output_matrix = []
        for i=1:out_size
            push!(output_matrix, GP(states_new, target_data[i,:], gp.mean, gp.kernel, gp.lognoise))
            gp.optimize ? optimize!(output_matrix[i]; method=gp.optimizer) : nothing
        end
    end
    OutputLayer(gp, output_matrix, out_size)
end