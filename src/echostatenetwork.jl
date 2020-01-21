function states_matrix(W::Matrix{Float64}, 
        W_in::Matrix{Float64}, 
        train_data::Matrix{Float64}, 
        alpha::Float64)
        
    train_len = size(train_data)[2]
    res_size = size(W)[1]    
    states = zeros(Float64, res_size, train_len)
    for i=1:train_len-1
        states[:, i+1] = (1-alpha).*states[:, i] + alpha*tanh.((W*states[:, i])+(W_in*train_data[:, i]))
    end
    return states
end

function esn_train(states::Matrix{Float64}, 
        train_data::Matrix{Float64}, 
        beta::Float64,  
        nonlin_alg::String)
        
    res_size = size(states)[1]
    i_mat = beta.*Matrix(1.0I, res_size, res_size)
    states_new = copy(states)
    if nonlin_alg == nothing
        states_new = states_new
    elseif nonlin_alg == "T1"
        for i=1:size(states_new, 1)
            if mod(i, 2)!=0
                states_new[i, :] = copy(states[i,:].*states[i,:])
            end
         end
    elseif nonlin_alg == "T2"
        for i=2:size(states_new, 1)-1
            if mod(i, 2)!=0
                states_new[i, :] = copy(states[i-1,:].*states[i-2,:])
            end
         end
    elseif nonlin_alg == "T3"
        for i=2:size(states_new, 1)-1
            if mod(i, 2)!=0
                states_new[i, :] = copy(states[i-1,:].*states[i+1,:])
            end
         end
    end
    W_out = (train_data*transpose(states_new))*inv(states_new*transpose(states_new)+i_mat)

    return W_out
end

function esn_predict(predict_len::Int, 
        W_in::Matrix{Float64},
        W::Matrix{Float64}, 
        W_out::Matrix{Float64}, 
        states::Matrix{Float64},
        alpha::Float64, 
        nonlin_alg::String)
        
    in_size = size(W_in)[2]
    output = zeros(Float64, in_size, predict_len)
    x = states[:, end]
    for i=1:predict_len
        x_new = copy(x)
        if nonlin_alg == nothing
            x_new = x_new
        elseif nonlin_alg == "T1"
            for j=1:size(x_new, 1)
                if mod(j, 2)!=0
                    x_new[j] = copy(x[j]*x[j])
                end
            end 
        elseif nonlin_alg == "T2"
            for j=2:size(x_new, 1)-1
                if mod(j, 2)!=0
                    x_new[j] = copy(x[j-1]*x[j-2])
                end
            end 
        elseif nonlin_alg == "T3"
            for j=2:size(x_new, 1)-1
                if mod(j, 2)!=0
                    x_new[j] = copy(x[j-1]*x[j+1])
                end
            end 
        end
        out = (W_out*x_new)
        output[:, i] = out
        x = (1-alpha).*x + alpha*tanh.((W*x)+(W_in*out))
    end
    return output
end
