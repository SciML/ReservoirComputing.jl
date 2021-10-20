struct GaussianRegression{M,K,L,O} <: GaussianModel
    mean::M
    kernel::K
    lognoise::L
    optimize::Bool
    optimizer::O
end

function GaussianRegression(mean, kernel;
                            lognoise=-2, 
                            optimize=false,
                            optimizer=Optim.LBFGS())
    GaussianRegression(mean, kernel, lognoise, optimize, optimizer)
end

function train!(rc::AbstractReservoirComputer, target_data, gr::GaussianRegression)

    states_new = nla(rc.nla_type, rc.states)
    if size(target_data, 1) == 1
        gp = GP(states_new, vec(target_data), mean, kernel, lognoise)
        optimize ? optimize!(gp; method=optimizer) : nothing
    else
        gp = []
        for i=1:size(target_data, 1)
            push!(gp, GP(states_new, target_data[i,:], mean, kernel, lognoise))
            optimize ? optimize!(gp[i]; method=optimizer) : nothing
        end
    end
    gp
end