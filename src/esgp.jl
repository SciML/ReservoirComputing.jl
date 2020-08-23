"""
    ESGPtrain(esn::AbstractLeakyESN, mean::GaussianProcesses.Mean, kernel::GaussianProcesses.Kernel 
    [, lognoise::Float64, optimize::Bool, optimizer::Optim.AbstractOptimizer, y_target::AbstractArray{Float64})

Train the ESN using Gaussian Processes, as described in [1]

[1] Chatzis, Sotirios P., and Yiannis Demiris. "Echo state Gaussian process." IEEE Transactions on Neural Networks 22.9 (2011): 1435-1445.
"""
function ESGPtrain(esn::AbstractLeakyESN,
    mean::GaussianProcesses.Mean, 
    kernel::GaussianProcesses.Kernel; 
    lognoise::Float64 = -2.0, 
    optimize::Bool = false,
    optimizer::Optim.AbstractOptimizer = Optim.LBFGS(),
    y_target::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    if size(y_target, 1) == 1
        gp = GP(states_new, vec(y_target), mean, kernel, lognoise)
        if optimize == true
            optimize!(gp; method=optimizer)
        end
    else
        gp = []
        for i=1:size(y_target, 1)
            push!(gp, GP(states_new, y_target[i,:], mean, kernel, lognoise))
            if optimize == true
                optimize!(gp[i]; method=optimizer)
            end
        end
    end
    return gp
end

#one variable timestep prediction
function ESGPpredict(esn::AbstractLeakyESN,
    predict_len::Int,
    gp::GaussianProcesses.GPE)
    
    output = zeros(Float64, esn.in_size, predict_len)
    sigmas = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)'
            out, sigma = GaussianProcesses.predict_y(gp, x_new)
            output[:, i] = out
            sigmas[:,i] = sigma
            x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
        end
    else
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)'
            out, sigma = GaussianProcesses.predict_y(gp, x_new)
            output[:, i] = out
            sigmas[:,i] = sigma
            x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out) 
        end
    end
    return output, sigmas
end

#multidimensional timestep prediction

"""
    ESGPpredict(esn::AbstractLeakyESN, predict_len::Int, gp::AbstractArray{Any})
    
Return the prediction for a given lenght of the constructed ESN struct using GPs.

"""
function ESGPpredict(esn::AbstractLeakyESN,
    predict_len::Int,
    gp::AbstractArray{Any})
    
    output = zeros(Float64, esn.in_size, predict_len)
    sigmas = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)'
            out = Array{Float64}(undef, esn.out_size)
            sigma = Array{Float64}(undef, esn.out_size)
            for i=1:size(gp, 1)
                out[i] = GaussianProcesses.predict_y(gp[i], x_new)[1][1]
                sigma[i] = GaussianProcesses.predict_y(gp[i], x_new)[2][1]
            end
            output[:, i] = out
            sigmas[:,i] = sigma
            x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
        end
    else
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)'
            out = Array{Float64}(undef, esn.out_size)
            sigma = Array{Float64}(undef, esn.out_size)
            for i=1:size(gp, 1)
                out[i] = GaussianProcesses.predict_y(gp[i], x_new)[1][1]
                sigma[i] = GaussianProcesses.predict_y(gp[i], x_new)[2][1]
            end
            output[:, i] = out
            sigmas[:,i] = sigma
            x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out) 
        end
    end
    return output, sigmas
end


"""
    ESGPpredict_h_steps(esn::AbstractLeakyESN, predict_len::Int, h_steps::Int, test_data::AbstractArray{Float64}, gp::GaussianProcesses.GPE)

Return the prediction for h steps ahead of the constructed ESN struct using GPs.
"""
function ESGPpredict_h_steps(esn::AbstractLeakyESN,
    predict_len::Int,
    h_steps::Int,
    test_data::AbstractArray{Float64},
    gp::GaussianProcesses.GPE)
    
    output = zeros(Float64, esn.in_size, predict_len)
    sigmas = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)'
            out, sigma = GaussianProcesses.predict_y(gp, x_new)
            output[:, i] = out
            sigmas[:,i] = sigma
            if mod(i, h_steps) == 0
                x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, test_data[:,i])
            else
                x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
            end
        end
    else
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)'
            out, sigma = GaussianProcesses.predict_y(gp, x_new)
            output[:, i] = out
            sigmas[:,i] = sigma
            if mod(i, h_steps) == 0
                x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], test_data[:,i]), test_data[:,i])
            else
                x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out)
            end
        end
    end
    return output, sigmas
end  
