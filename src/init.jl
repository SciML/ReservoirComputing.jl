function init_reservoir(approx_res_size::Int, 
        in_size::Int,
        radius::Float64, 
        degree::Int)
    
    res_size = Int(floor(approx_res_size/in_size)*in_size)
    sparsity = degree/res_size
    W = Matrix(sprand(Float64, res_size, res_size, sparsity))
    W = 2.0 .*(W.-0.5)
    replace!(W, -1.0=>0.0)
    rho_w = maximum(abs.(eigvals(W)))
    W .*= radius/rho_w
    return W
end

function init_input_layer(approx_res_size::Int, 
        in_size::Int, 
        sigma::Float64)
    res_size = Int(floor(approx_res_size/in_size)*in_size)
    W_in = zeros(Float64, res_size, in_size)
    q = Int(res_size/in_size)
    for i=1:in_size
        W_in[(i-1)*q+1 : (i)*q, i] = (2*sigma).*(rand(Float64, 1, q).-0.5)
    end
    return W_in
end 
