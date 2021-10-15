abstract type AbstractReservoir end

struct RandSparseReservoir{T,C} <: AbstractReservoir
    radius::T
    sparsity::C
end

function RandSparseReservoir(;radius=1.0, sparsity::Float64=0.1)
    RandSparseReservoir(radius, sparsity)
end

function create_reservoir(res_size, reservoir::RandSparseReservoir)
    reservoir_matrix = Matrix(sprand(Float64, res_size, res_size, reservoir.sparsity))
    reservoir_matrix = 2.0 .*(reservoir_matrix.-0.5)
    replace!(reservoir_matrix, -1.0=>0.0)
    rho_w = maximum(abs.(eigvals(W)))
    reservoir_matrix .*= reservoir.radius/rho_w
    reservoir_matrix
end

#=
function create_reservoir(res_size, reservoir::RandReservoir)
    sparsity = degree/res_size
    W = Matrix(sprand(Float64, res_size, res_size, sparsity))
    W = 2.0 .*(W.-0.5)
    replace!(W, -1.0=>0.0)
    rho_w = maximum(abs.(eigvals(W)))
    W .*= radius/rho_w
    W
end
=#    


#SVD reservoir construction based on "Yang, Cuili, et al. "Design of polynomial echo state networks for time series prediction" Yang et al

"""
    pseudoSVD(dim::Int,  max_value::Float64, sparsity::Float64 [, sorted::Bool, reverse_sort::Bool])

Return a reservoir matrix created using SVD as described in [1].

[1] Yang, Cuili, et al. "Design of polynomial echo state networks for time series prediction." Neurocomputing 290 (2018): 148-160.
"""

struct PseudoSVDReservoir{T,C} <: AbstractReservoir
    max_value::T
    sparsity::C
    sorted::Bool
    reverse_sort::Bool
end

function PseudoSVDReservoir(;max_value=1.0, sparsity=0.1, sorted=true, reverse_sort=false)
    PseudoSVDReservoir(max_value, sparsity, sorted, reverse_sort)
end

function create_reservoir(res_size, reservoir::PseudoSVDReservoir)
    reservoir_matrix = create_diag(res_size, reservoir.max_value, sorted = reservoir.sorted, reverse_sort = reservoir.reverse_sort)
    tmp_sparsity = get_sparsity(reservoir_matrix, res_size)

    while tmp_sparsity <= sparsity
        reservoir_matrix *= create_qmatrix(res_size, rand(1:res_size), rand(1:res_size), rand()*2-1)
        tmp_sparsity = get_sparsity(reservoir_matrix, res_size)
    end
    reservoir_matrix
end

function create_diag(dim, max_value; sorted = true, reverse_sort = false)

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
    diagonal_matrix
end

function create_qmatrix(dim, coord_i, coord_j, theta)

    qmatrix = zeros(Float64, dim, dim)
    for i = 1:dim
        qmatrix[i,i] = 1.0
    end
    qmatrix[coord_i, coord_i] = cos(theta)
    qmatrix[coord_j, coord_j] = cos(theta)
    qmatrix[coord_i, coord_j] = -sin(theta)
    qmatrix[coord_j, coord_i] = sin(theta)

    qmatrix
end

function get_sparsity(M, dim)
    size(M[M .!= 0], 1)/(dim*dim-size(M[M .!= 0], 1)) #nonzero/zero elements
end

#from "minimum complexity echo state network" Rodan
# Delay Line Reservoir
"""
    DLR(res_size::Int, weight::Float64)

Return a Delay Line Reservoir matrix as described in [2].

[2] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
"""

struct DelayLineReservoir{T} <: AbstractReservoir
    weight::T
end

function DelayLineReservoir(;weight=0.1)
    DelayLineReservoir(weight)
end

function create_reservoir(res_size, reservoir::DelayLineReservoir)

    reservoir_matrix = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        reservoir_matrix[i+1,i] = reservoir.weight
    end
    reservoir_matrix
end

#from "minimum complexity echo state network" Rodan
# Delay Line Reservoir with backward connections

"""
    DLRB(res_size::Int, weight::Float64, fb_weight::Float64)

Return a Delay Line Reservoir matrix with Backward connections as described in [2].

[2] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
"""

struct DelayLineBackwardReservoir{T} <: AbstractReservoir
    weight::T
    fb_weight::T
end

function DelayLineBackwardReservoir(;weight=0.1, fb_weight=0.2)
    DelayLineBackwardReservoir(weight, fb_weight)
end

function create_reservoir(res_size, reservoir::DelayLineBackwardReservoir)

    reservoir_matrix = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        reservoir_matrix[i+1,i] = reservoir.weight
        reservoir_matrix[i,i+1] = reservoir.fb_weight
    end
    reservoir_matrix
end

#from "minimum complexity echo state network" Rodan
# Simple cycle reservoir
"""
    SCR(res_size::Int, weight::Float64)

Return a Simple Cycle Reservoir Reservoir matrix as described in [2].

[2] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
"""

struct SimpleCycleReservoir{T} <: AbstractReservoir
    weight::T
end

function SimpleCycleReservoir(;weight=0.1)
    SimpleCycleReservoir(weight)
end

function create_reservoir(res_size, reservoir::SimpleCycleReservoir)

    reservoir_matrix = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        reservoir_matrix[i+1,i] = reservoir.weight
    end
    reservoir_matrix[1, res_size] = reservoir.weight
    reservoir_matrix
end

#from "simple deterministically constructed cycle reservoirs with regular jumps" by Rodan and Tino
# Cycle Reservoir with Jumps

"""
    CRJ(res_size::Int, cycle_weight::Float64, jump_weight::Float64, jump_size::Int)

Return a Cycle Reservoir with Jumps matrix as described in [2].

[2] Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""

struct CycleJumpsReservoir{T,C} <: AbstractReservoir
    cycle_weight::T
    jump_weight::T
    jump_size::C
end

function CycleJumpsReservoir(;cycle_weight=0.1, jump_weight=0.1,jump_size=3)
    CycleJumpsReservoir(cycle_weight, jump_weight, jump_size)
end

function create_reservoir(res_size, reservoir::CycleJumpsReservoir)

    reservoir_matrix = zeros(Float64, res_size, res_size)
    for i=1:res_size-1
        reservoir_matrix[i+1,i] = reservoir.cycle_weight
    end
    reservoir_matrix[1, res_size] = reservoir.cycle_weight

    for i=1:reservoir.jump_size:res_size-reservoir.jump_size
        tmp = (i+reservoir.jump_size)%res_size

        if tmp == 0
            tmp = res_size
        end

        reservoir_matrix[i, tmp] = reservoir.jump_weight
        reservoir_matrix[tmp, i] = reservoir.jump_weight
    end
    reservoir_matrix
end
