struct GaussianProcess{M,K,L,O} <: AbstractGaussianProcess
    mean::M
    kernel::K
    lognoise::L
    optimize::Bool
    optimizer::O
end

"""
    GaussianProcess(mean, kernel;
        lognoise=-2, 
        optimize=false,
        optimizer=Optim.LBFGS())

Wrapper around [GaussianProcesses](http://stor-i.github.io/GaussianProcesses.jl/latest/) gives the possibility of 
training every model in the library using Gaussian Regression. GaussianProcesses.jl needs to be called in order to use these 
models. The use of Gaussian Regression for ESNs has first been explored in [1].

[1] Chatzis, Sotirios P., and Yiannis Demiris. "_Echo state Gaussian process._" IEEE Transactions on Neural Networks 22.9 (2011): 1435-1445.
"""
function GaussianProcess(mean, kernel;
                            lognoise=-2, 
                            optimize=false,
                            optimizer=Optim.LBFGS())
    GaussianProcess(mean, kernel, lognoise, optimize, optimizer)
end

function _train(states, target_data, gp::GaussianProcess)

    out_size = size(target_data, 1)
    output_matrix = []
    for i=1:out_size
        out_size == 1 ? target = vec(target_data) : target = target_data[i,:]
        push!(output_matrix, GP(states, target, gp.mean, gp.kernel, gp.lognoise))
        gp.optimize ? optimize!(output_matrix[i]; method=gp.optimizer) : nothing
    end
    OutputLayer(gp, output_matrix, out_size, target_data[:,end])
end