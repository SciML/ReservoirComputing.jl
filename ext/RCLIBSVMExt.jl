module RCLIBSVMExt

using LIBSVM
using ReservoirComputing:
                          SVMReadout, addreadout!, ReservoirChain
import ReservoirComputing: train

function train(svr::LIBSVM.AbstractSVR,
        states::AbstractArray, target::AbstractArray)
    @assert size(states, 2)==size(target, 2) "states and target must share columns."
    perm_states = permutedims(states)
    size_target = size(target, 1)

    if size_target == 1
        vec_target = vec(target)
        model = LIBSVM.fit!(svr, perm_states, vec_target)
        return model
    else
        models = Vector{Any}(undef, size_target)
        for (idx, row_target) in enumerate(eachrow(target))
            models[idx] = LIBSVM.fit!(svr, perm_states, row_target)
        end
        return models
    end
end

_has_models(ps) = (ps isa NamedTuple) && (:models in propertynames(ps))

function (svmro::SVMReadout)(inp::AbstractArray, ps, st::NamedTuple)
    if !_has_models(ps)
        return inp, st
    end
    models = getfield(ps, :models)

    vec_like = false
    if ndims(inp) == 1
        reshaped_inp = reshape(inp, 1, :)
        num_imp = 1
        vec_like = true
    elseif ndims(inp) == 2
        if size(inp, 2) == 1
            reshaped_inp = reshape(vec(inp), 1, :)
            num_inp = 1
            vec_like = true
        else
            reshaped_inp = permutedims(inp)
            num_imp = size(reshaped_inp, 1)
        end
    else
        throw(ArgumentError("SVMReadout expects 1D or 2D input; got size $(size(inp))"))
    end

    if models isa AbstractVector
        out_data = Array{float(eltype(reshaped_inp))}(undef, svmro.out_dims, num_imp)
        for (idx, model) in enumerate(models)
            single_out = LIBSVM.predict(models[idx], reshaped_inp)
            out_data[idx, :] = single_out
        end
    else
        single_out = LIBSVM.predict(models, reshaped_inp)
        out_data = reshape(single_out, 1, :)
    end

    if vec_like
        return vec(out_data), st
    else
        return out_data, st
    end
end

end # module
