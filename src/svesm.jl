function SVESMtrain(svr::LIBSVM.AbstractSVR,
    esn::AbstractLeakyESN; 
    y_target::AbstractArray{Float64} = esn.train_data)
    
    states_new = nla(esn.nla_type, esn.states)
    
    if size(y_target, 1) == 1
        fitted_svr = LIBSVM.fit!(svr, states_new', vec(y_target))
    else
        fitted_svr = []
        for i=1:size(y_target, 1)
            push!(fitted_svr, LIBSVM.fit!(svr, states_new', y_target[i,:]))
        end
    end
    return fitted_svr
end

function SVESM_direct_predict(esn::AbstractLeakyESN, 
    test_in::AbstractArray{Float64}, 
    fitted_svr::LIBSVM.AbstractSVR)
    x = esn.states[:, end]
    prediction_states = zeros(Float64, size(esn.states, 1), size(test_in, 2))
    
    if esn.extended_states == false
        for i=1:size(test_in, 2)
            x_new = nla(esn.nla_type, x)
            x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x_new, test_in[:,i])
            prediction_states[:, i] = x
        end
    else
        for i=1:size(test_in, 2)
            x_new = nla(esn.nla_type, x)
            x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x_new[1:esn.res_size], test_in[:,i]), test_in[:, i]) 
            prediction_states[:,i] = x
        end
    end
    output = LIBSVM.predict(fitted_svr, prediction_states')
    return output
end 

#predict if one one variable timeseries is provided
function SVESMpredict(esn::AbstractLeakyESN, 
    predict_len::Int, 
    fitted_svr::LIBSVM.AbstractSVR)
    
    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]
    if esn.extended_states == false
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)
            out = LIBSVM.predict(fitted_svr, x_new)
            output[:, i] = out
            x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
        end
    else
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)
            out = LIBSVM.predict(fitted_svr, x_new)
            output[:, i] = out
            x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out)
        end
    end
    return output
end

#predict for multidimensional timeseries
function SVESMpredict(esn::AbstractLeakyESN, 
    predict_len::Int, 
    fitted_svr::AbstractArray{Any})
    
    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]
    
    if esn.extended_states == false
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)
            out = []
            for i=1:size(fitted_svr, 1)
                push!(out, LIBSVM.predict(fitted_svr[i], x_new)[1])
            end
            output[:, i] = out
            x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
        end
    else
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)
            out = Array{Float64}(undef, esn.out_size)
            for i=1:size(fitted_svr, 1)
                out[i] = LIBSVM.predict(fitted_svr[i], x_new)[1]
            end
            output[:, i] = out
            x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out)
        end
    end
    return output
end

function SVESMpredict_h_steps(esn::AbstractLeakyESN, 
    predict_len::Int,  
    h_steps::Int, 
    test_data::AbstractArray{Float64}, 
    fitted_svr::LIBSVM.AbstractSVR)
    
    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]
    if esn.extended_states == false
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)
            out = LIBSVM.predict(fitted_svr, x_new)
            output[:, i] = out
            if mod(i, h_steps) == 0
                x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, test_data[:,i])
            else
                x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
            end
        end
    else
        for i=1:predict_len
            x_new = hcat(nla(esn.nla_type, x)...)
            out = LIBSVM.predict(fitted_svr, x_new)
            output[:, i] = out
            if mod(i, h_steps) == 0
                x = vcat(ReservoirComputing.leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], test_data[:,i]), test_data[:,i])
            else
                x = vcat(ReservoirComputing.leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out)
            end
        end
    end
    return output
end
