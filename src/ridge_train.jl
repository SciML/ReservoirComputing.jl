function ESNtrain(esn::AbstractEchoStateNetwork, beta::Float64)
    
    states_new = nla(esn.nla_type, esn.states)
    W_out = (esn.train_data*states_new')*inv(add_reg(states_new*states_new', beta))

    return W_out
end

function add_reg(X::AbstractArray{Float64}, beta::Float64)
    n = size(X, 1)
    for i=1:n
        X[i,i] += beta
    end
    return X
end 
