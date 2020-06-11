
function SVESMtrain(svr::LIBSVM.AbstractSVR,
    esn::AbstractLeakyESN, 
    y_target::AbstractArray{Float64})
    
    states_new = ReservoirComputing.nla(esn.nla_type, esn.states)
    m = LIBSVM.fit!(svr, states_new', y_target)
    return m
end

function SVESM_direct_predict(esn::AbstractLeakyESN, 
    test_in::AbstractArray{Float64}, 
    m::LIBSVM.AbstractSVR)
    x = esn.states[:, end]
    prediction_states = zeros(Float64, size(esn.states, 1), size(test_in, 2))
    
    if esn.extended_states == false
        for i=1:size(test_in, 2)
            x_new = nla(esn.nla_type, x)
            x = (1-esn.alpha).*x + esn.alpha*esn.activation.((esn.W*x)+(esn.W_in*test_in[:,i]))
            prediction_states[:, i] = x
        end
    else
        for i=1:size(test_in, 2)
            x_new = nla(esn.nla_type, x)
            x = vcat((1-esn.alpha).*x[1:esn.res_size] + esn.alpha*esn.activation.((esn.W*x[1:esn.res_size])+(esn.W_in*test_in[:, i])), 
                test_in[:, i]) 
            prediction_states[:,i] = x
        end
    end
    output = LIBSVM.predict(m, prediction_states')
    return output
end 
