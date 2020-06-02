function init_input_layer(res_size::Int,
        in_size::Int,
        sigma::Float64)

    W_in = zeros(Float64, res_size, in_size)
    q = Int(res_size/in_size)
    for i=1:in_size
        W_in[(i-1)*q+1 : (i)*q, i] = (2*sigma).*(rand(Float64, 1, q).-0.5)
    end
    return W_in

end 

function init_dense_input_layer(res_size::Int,
        in_size::Int,
        sigma::Float64)

    W_in = rand(Float64, res_size, in_size)
    W_in = 2.0 .*(W_in.-0.5)
    W_in = sigma .*W_in
    return W_in
end

function init_sparse_input_layer(res_size::Int,
        in_size::Int,
        sigma::Float64,
        sparsity::Float64)

    W_in = Matrix(sprand(Float64, res_size, in_size, sparsity))
    W_in = 2.0 .*(W_in.-0.5)
    replace!(W_in, -1.0=>0.0)
    W_in = sigma .*W_in
    return W_in
end
