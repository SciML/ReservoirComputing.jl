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

function train!(rc::AbstractReservoirComputer, target_data, gp::GaussianProcess)

    states_new = nla(rc.nla_type, rc.states)
    if size(target_data, 1) == 1
        output_layer = GP(states_new, vec(target_data), gp.mean, gp.kernel, gp.lognoise)
        gp.optimize ? optimize!(output_layer; method=gp.optimizer) : nothing
    else
        output_layer = []
        for i=1:size(target_data, 1)
            push!(output_layer, GP(states_new, target_data[i,:], gp.mean, gp.kernel, gp.lognoise))
            gp.optimize ? optimize!(output_layer[i]; method=gp.optimizer) : nothing
        end
    end
    output_layer
end