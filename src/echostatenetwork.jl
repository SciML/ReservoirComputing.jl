struct ESN{T<:AbstractFloat}
    res_size::Integer
    in_size::Integer
    out_size::Integer
    train_data::Matrix{T}
    degree::Integer
    sigma::T
    alpha::T
    beta::T
    radius::T
    nonlin_alg::String
    W::Matrix{T}
    W_in::Matrix{T}
    states::Matrix{T}
    
    function ESN(approx_res_size::Integer,
            in_size::Integer, 
            out_size::Integer, 
            train_data::Matrix{T}, 
            degree::Integer, 
            sigma::T,
            alpha::T, 
            beta::T,
            radius::T,
            nonlin_alg::String) where T<:AbstractFloat

        res_size = Int(floor(approx_res_size/in_size)*in_size)
        W = init_reservoir(res_size, in_size, radius, degree)
        W_in = init_input_layer(res_size, in_size, sigma)
        states = states_matrix(W, W_in, train_data, alpha)
        
        return new{T}(res_size, in_size, out_size, train_data, 
        degree, sigma, alpha, beta, radius, nonlin_alg, W, W_in, states)
    end
end


function init_reservoir(res_size::Integer, 
        in_size::Integer,
        radius::Float64, 
        degree::Integer)
    
    sparsity = degree/res_size
    W = Matrix(sprand(Float64, res_size, res_size, sparsity))
    W = 2.0 .*(W.-0.5)
    replace!(W, -1.0=>0.0)
    rho_w = maximum(abs.(eigvals(W)))
    W .*= radius/rho_w
    return W
end

function init_input_layer(res_size::Integer, 
        in_size::Integer, 
        sigma::Float64)
        
    W_in = zeros(Float64, res_size, in_size)
    q = Integer(res_size/in_size)
    for i=1:in_size
        W_in[(i-1)*q+1 : (i)*q, i] = (2*sigma).*(rand(Float64, 1, q).-0.5)
    end
    return W_in
end 

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

function ESNtrain(esn::ESN)
    
    i_mat = esn.beta.*Matrix(1.0I, esn.res_size, esn.res_size)
    states_new = copy(esn.states)
    if esn.nonlin_alg == nothing
        states_new = states_new
    elseif esn.nonlin_alg == "T1"
        for i=1:size(states_new, 1)
            if mod(i, 2)!=0
                states_new[i, :] = copy(esn.states[i,:].*esn.states[i,:])
            end
         end
    elseif esn.nonlin_alg == "T2"
        for i=2:size(states_new, 1)-1
            if mod(i, 2)!=0
                states_new[i, :] = copy(esn.states[i-1,:].*esn.states[i-2,:])
            end
         end
    elseif esn.nonlin_alg == "T3"
        for i=2:size(states_new, 1)-1
            if mod(i, 2)!=0
                states_new[i, :] = copy(esn.states[i-1,:].*esn.states[i+1,:])
            end
         end
    end
    W_out = (esn.train_data*transpose(states_new))*inv(states_new*transpose(states_new)+i_mat)

    return W_out
end

function ESNpredict(esn::ESN, 
    predict_len::Integer,
    W_out::Matrix{Float64})
    
    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]
    for i=1:predict_len
        x_new = copy(x)
        if esn.nonlin_alg == nothing
            x_new = x_new
        elseif esn.nonlin_alg == "T1"
            for j=1:size(x_new, 1)
                if mod(j, 2)!=0
                    x_new[j] = copy(x[j]*x[j])
                end
            end 
        elseif esn.nonlin_alg == "T2"
            for j=2:size(x_new, 1)-1
                if mod(j, 2)!=0
                    x_new[j] = copy(x[j-1]*x[j-2])
                end
            end 
        elseif esn.nonlin_alg == "T3"
            for j=2:size(x_new, 1)-1
                if mod(j, 2)!=0
                    x_new[j] = copy(x[j-1]*x[j+1])
                end
            end 
        end
        out = (W_out*x_new)
        output[:, i] = out
        x = (1-esn.alpha).*x + esn.alpha*tanh.((esn.W*x)+(esn.W_in*out))
    end
    return output
end
