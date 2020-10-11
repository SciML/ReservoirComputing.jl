 #given degree of connections between neurons
 """
     init_reservoir_givendeg(res_size::Int, radius::Float64, degree::Int)

Return a reservoir matrix scaled by the radius value and with a given degree of connection.
 """
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
 """
     init_reservoir_givensp(res_size::Int, radius::Float64, sparsity::Float64)

Return a reservoir matrix scaled by the radius value and with a given sparsity.
 """
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

#SVD reservoir construction based on "Yang, Cuili, et al. "Design of polynomial echo state networks for time series prediction" Yang et al

"""
    pseudoSVD(dim::Int,  max_value::Float64, sparsity::Float64 [, sorted::Bool, reverse_sort::Bool])

Return a reservoir matrix created using SVD as described in [1].

[1] Yang, Cuili, et al. "Design of polynomial echo state networks for time series prediction." Neurocomputing 290 (2018): 148-160.
"""
function pseudoSVD(dim::Int,
        max_value::Float64,
        sparsity::Float64;
        sorted::Bool = true,
        reverse_sort::Bool = false)

    S = create_diag(dim, max_value, sorted = sorted, reverse_sort = reverse_sort)
    sp = get_sparsity(S, dim)

    while sp <= sparsity
        S *= create_qmatrix(dim, rand(1:dim), rand(1:dim), rand(Float64)*2-1)
        sp = get_sparsity(S, dim)
    end
    return S
end

function create_diag(dim::Int,
        max_value::Float64;
        sorted::Bool = true,
        reverse_sort::Bool = false)

    diagonal_matrix = zeros(Float64, dim, dim)
    if sorted == true
        if reverse_sort == true
            diagonal_values = sort(rand(Float64, dim).*max_value, rev = true)
            diagonal_values[1] = max_value
        else
            diagonal_values = sort(rand(Float64, dim).*max_value)
            diagonal_values[end] = max_value
        end
    else
        diagonal_values = rand(Float64, dim).*max_value
    end

    for i=1:dim
        diagonal_matrix[i, i] = diagonal_values[i]
    end
    return diagonal_matrix
end

function create_qmatrix(dim::Int,
        coord_i::Int,
        coord_j::Int,
        theta::Float64)

    qmatrix = zeros(Float64, dim, dim)
    for i = 1:dim
        qmatrix[i,i] = 1.0
    end
    qmatrix[coord_i, coord_i] = cos(theta)
    qmatrix[coord_j, coord_j] = cos(theta)
    qmatrix[coord_i, coord_j] = -sin(theta)
    qmatrix[coord_j, coord_i] = sin(theta)

    return qmatrix
end

function get_sparsity(M::AbstractArray{Float64}, dim::Int)
    return size(M[M .!= 0], 1)/(dim*dim-size(M[M .!= 0], 1)) #nonzero/zero elements
end

#from "minimum complexity echo state network" Rodan
# Delay Line Reservoir
"""
    DLR(res_size::Int, weight::Float64)

Return a Delay Line Reservoir matrix as described in [2].

[2] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function DLR(res_size::Int,
        weight::Float64)

    W = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        W[i+1,i] = weight
    end
    return W
end

#from "minimum complexity echo state network" Rodan
# Delay Line Reservoir with backward connections

"""
    DLRB(res_size::Int, weight::Float64, fb_weight::Float64)

Return a Delay Line Reservoir matrix with Backward connections as described in [2].

[2] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function DLRB(res_size::Int,
        weight::Float64,
        fb_weight::Float64)

    W = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        W[i+1,i] = weight
        W[i,i+1] = fb_weight
    end
    return W
end

#from "minimum complexity echo state network" Rodan
# Simple cycle reservoir
"""
    SCR(res_size::Int, weight::Float64)

Return a Simple Cycle Reservoir Reservoir matrix as described in [2].

[2] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function SCR(res_size::Int,
        weight::Float64)

    W = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        W[i+1,i] = weight
    end
    W[1, res_size] = weight
    return W
end

#from "simple deterministically constructed cycle reservoirs with regular jumps" by Rodan and Tino
# Cycle Reservoir with Jumps

"""
    CRJ(res_size::Int, cycle_weight::Float64, jump_weight::Float64, jump_size::Int)

Return a Cycle Reservoir with Jumps matrix as described in [2].

[2] Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""
function CRJ(res_size::Int,
        cycle_weight::Float64,
        jump_weight::Float64,
        jump_size::Int)

    W = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        W[i+1,i] = cycle_weight
    end
    W[1, res_size] = cycle_weight

    for i=1:jump_size:res_size-jump_size
        tmp = (i+jump_size)%res_size

        if tmp == 0
            tmp = res_size
        end

        W[i, tmp] = jump_weight
        W[tmp, i] = jump_weight
    end
    return W
end
