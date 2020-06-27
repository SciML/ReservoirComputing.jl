 #given degree of connections between neurons
 function init_reservoir_givendeg(res_size::Int,
        radius::Float64,
        degree::Int)

    sparsity = degree/res_size
    W = Matrix(sprand(Float64, res_size, res_size, sparsity))
    W = 2.0 .*(W.-0.5)
    replace!(W, -1.0=>0.0)
    rho_w = maximum(abs.(eigvals(W)))
    W .*= radius/rho_w
    return W
end

#given sparsity of connection between neurons
function init_reservoir_givensp(res_size::Int,
        radius::Float64,
        sparsity::Float64)

    W = Matrix(sprand(Float64, res_size, res_size, sparsity))
    W = 2.0 .*(W.-0.5)
    replace!(W, -1.0=>0.0)
    rho_w = maximum(abs.(eigvals(W)))
    W .*= radius/rho_w
    return W
end


function pseudoSVD(dim::Int, 
        max_value::Float64, 
        sparsity::Float64)
    
    S = create_diag(dim, max_value)
    sp = get_sparsity(S)
    
    while sp < sparsity
        S *= create_qmatrix(dim, rand(1:dim), rand(1:dim), rand(Float64)*2-1)
        sp = get_sparsity(S)
    end
    return S
end

function create_diag(dim::Int, 
        max_value::Float64; 
        sorted::Bool=true)
    
    if sorted == true
        diagonal_values = sort(rand(Float64, dim).*max_value)
        diagonal_values[end] = max_value
    else
        diagonal_values = rand(Float64, dim).*max_value
    end
    
    return diagm(0 => diagonal_values)
end

function create_qmatrix(dim::Int, 
        coord_i::Int, 
        coord_j::Int, 
        theta::Float64)
    
    qmatrix = diagm(ones(Float64, dim))
    qmatrix[coord_i, coord_i] = cos(theta)
    qmatrix[coord_j, coord_j] = cos(theta)
    qmatrix[coord_i, coord_j] = -sin(theta)
    qmatrix[coord_j, coord_i] = sin(theta)
    
    return qmatrix
end

function get_sparsity(M::AbstractArray{Float64})
    return size(M[M .!= 0], 1)/(size(M, 1)*size(M, 1)-size(M[M .!= 0], 1)) #nonzero/zero elements
end
