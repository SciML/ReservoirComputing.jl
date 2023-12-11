abstract type AbstractReservoir end

function get_ressize(reservoir::AbstractReservoir)
    return reservoir.res_size
end

function get_ressize(reservoir)
    return size(reservoir, 1)
end

struct RandSparseReservoir{T, C} <: AbstractReservoir
    res_size::Int
    radius::T
    sparsity::C
end

"""
    RandSparseReservoir(res_size, radius, sparsity)
    RandSparseReservoir(res_size; radius=1.0, sparsity=0.1)


Returns a random sparse reservoir initializer, which generates a matrix of size `res_size x res_size` with the specified `sparsity` and scaled spectral radius according to `radius`. This type of reservoir initializer is commonly used in Echo State Networks (ESNs) for capturing complex temporal dependencies.

# Arguments
- `res_size`: The size of the reservoir matrix.
- `radius`: The desired spectral radius of the reservoir. By default, it is set to 1.0.
- `sparsity`: The sparsity level of the reservoir matrix, controlling the fraction of zero elements. By default, it is set to 0.1.

# Returns
A RandSparseReservoir object that can be used as a reservoir initializer in ESN construction.

# References
This type of reservoir initialization is a common choice in ESN construction for its ability to capture temporal dependencies in data. However, there is no specific reference associated with this function.
"""
function RandSparseReservoir(res_size; radius = 1.0, sparsity = 0.1)
    return RandSparseReservoir(res_size, radius, sparsity)
end

"""
    create_reservoir(reservoir::AbstractReservoir, res_size)
    create_reservoir(reservoir, args...)

Given an `AbstractReservoir` constructor and the size of the reservoir (`res_size`), this function returns the corresponding reservoir matrix. Alternatively, it accepts a pre-generated matrix.

# Arguments
- `reservoir`: An `AbstractReservoir` object or constructor.
- `res_size`: The size of the reservoir matrix.
- `matrix_type`: The type of the resulting matrix. By default, it is set to `Matrix{Float64}`.

# Returns
A matrix representing the reservoir, generated based on the properties of the specified `reservoir` object or constructor.

# References
The choice of reservoir initialization is crucial in Echo State Networks (ESNs) for achieving effective temporal modeling. Specific references for reservoir initialization methods may vary based on the type of reservoir used, but the practice of initializing reservoirs for ESNs is widely documented in the ESN literature.
"""
function create_reservoir(reservoir::RandSparseReservoir,
                          res_size;
                          matrix_type = Matrix{Float64})
    reservoir_matrix = Matrix(sprand(res_size, res_size, reservoir.sparsity))
    reservoir_matrix = 2.0 .* (reservoir_matrix .- 0.5)
    replace!(reservoir_matrix, -1.0 => 0.0)
    rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= reservoir.radius / rho_w
    #TODO: change to explicit if
    Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix) ?
    error("Sparsity too low for size of the matrix.
          Increase res_size or increase sparsity") : nothing
    return Adapt.adapt(matrix_type, reservoir_matrix)
end

function create_reservoir(reservoir, args...; kwargs...)
    return reservoir
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

struct PseudoSVDReservoir{T, C} <: AbstractReservoir
    res_size::Int
    max_value::T
    sparsity::C
    sorted::Bool
    reverse_sort::Bool
end

function PseudoSVDReservoir(res_size;
                            max_value = 1.0,
                            sparsity = 0.1,
                            sorted = true,
                            reverse_sort = false)
    return PseudoSVDReservoir(res_size, max_value, sparsity, sorted, reverse_sort)
end

"""
    PseudoSVDReservoir(max_value, sparsity, sorted, reverse_sort)
    PseudoSVDReservoir(max_value, sparsity; sorted=true, reverse_sort=false)

Returns an initializer to build a sparse reservoir matrix with the given `sparsity` by using a pseudo-SVD approach as described in [^yang].

# Arguments
- `res_size`: The size of the reservoir matrix.
- `max_value`: The maximum absolute value of elements in the matrix.
- `sparsity`: The desired sparsity level of the reservoir matrix.
- `sorted`: A boolean indicating whether to sort the singular values before creating the diagonal matrix. By default, it is set to `true`.
- `reverse_sort`: A boolean indicating whether to reverse the sorted singular values. By default, it is set to `false`.

# Returns
A PseudoSVDReservoir object that can be used as a reservoir initializer in ESN construction.

# References
This reservoir initialization method, based on a pseudo-SVD approach, is inspired by the work in [^yang], which focuses on designing polynomial echo state networks for time series prediction.

[^yang]: Yang, Cuili, et al. "_Design of polynomial echo state networks for time series prediction._" Neurocomputing 290 (2018): 148-160.
"""
function PseudoSVDReservoir(res_size, max_value, sparsity; sorted = true,
                            reverse_sort = false)
    return PseudoSVDReservoir(res_size, max_value, sparsity, sorted, reverse_sort)
end

function create_reservoir(reservoir::PseudoSVDReservoir,
                          res_size;
                          matrix_type = Matrix{Float64})
    sorted = reservoir.sorted
    reverse_sort = reservoir.reverse_sort
    reservoir_matrix = create_diag(res_size, reservoir.max_value, sorted = sorted,
                                   reverse_sort = reverse_sort)
    tmp_sparsity = get_sparsity(reservoir_matrix, res_size)

    while tmp_sparsity <= reservoir.sparsity
        reservoir_matrix *= create_qmatrix(res_size, rand(1:res_size), rand(1:res_size),
                                           rand() * 2 - 1)
        tmp_sparsity = get_sparsity(reservoir_matrix, res_size)
    end

    return Adapt.adapt(matrix_type, reservoir_matrix)
end

function create_diag(dim, max_value; sorted = true, reverse_sort = false)
    diagonal_matrix = zeros(dim, dim)
    if sorted == true
        if reverse_sort == true
            diagonal_values = sort(rand(dim) .* max_value, rev = true)
            diagonal_values[1] = max_value
        else
            diagonal_values = sort(rand(dim) .* max_value)
            diagonal_values[end] = max_value
        end
    else
        diagonal_values = rand(dim) .* max_value
    end

    for i in 1:dim
        diagonal_matrix[i, i] = diagonal_values[i]
    end

    return diagonal_matrix
end

function create_qmatrix(dim, coord_i, coord_j, theta)
    qmatrix = zeros(dim, dim)

    for i in 1:dim
        qmatrix[i, i] = 1.0
    end

    qmatrix[coord_i, coord_i] = cos(theta)
    qmatrix[coord_j, coord_j] = cos(theta)
    qmatrix[coord_i, coord_j] = -sin(theta)
    qmatrix[coord_j, coord_i] = sin(theta)
    return qmatrix
end

function get_sparsity(M, dim)
    return size(M[M .!= 0], 1) / (dim * dim - size(M[M .!= 0], 1)) #nonzero/zero elements
end

#from "minimum complexity echo state network" Rodan
# Delay Line Reservoir

struct DelayLineReservoir{T} <: AbstractReservoir
    res_size::Int
    weight::T
end

"""
    DelayLineReservoir(res_size, weight)
    DelayLineReservoir(res_size; weight=0.1)

Returns a Delay Line Reservoir matrix constructor to obtain a deterministic reservoir as
described in [^Rodan2010].

# Arguments
- `res_size::Int`: The size of the reservoir.
- `weight::T`: The weight determines the absolute value of all the connections in the reservoir.

# Returns
A `DelayLineReservoir` object.

# References
[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function DelayLineReservoir(res_size; weight = 0.1)
    return DelayLineReservoir(res_size, weight)
end

function create_reservoir(reservoir::DelayLineReservoir,
                          res_size;
                          matrix_type = Matrix{Float64})
    reservoir_matrix = zeros(res_size, res_size)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = reservoir.weight
    end

    return Adapt.adapt(matrix_type, reservoir_matrix)
end

#from "minimum complexity echo state network" Rodan
# Delay Line Reservoir with backward connections
struct DelayLineBackwardReservoir{T} <: AbstractReservoir
    res_size::Int
    weight::T
    fb_weight::T
end

"""
    DelayLineBackwardReservoir(res_size, weight, fb_weight)
    DelayLineBackwardReservoir(res_size; weight=0.1, fb_weight=0.2)

Returns a Delay Line Reservoir constructor to create a matrix with backward connections
as described in [^Rodan2010]. The `weight` and `fb_weight` can be passed as either arguments or
keyword arguments, and they determine the absolute values of the connections in the reservoir.

# Arguments
- `res_size::Int`: The size of the reservoir.
- `weight::T`: The weight determines the absolute value of forward connections in the reservoir.
- `fb_weight::T`: The `fb_weight` determines the absolute value of backward connections in the reservoir.

# Returns
A `DelayLineBackwardReservoir` object.

# References
[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function DelayLineBackwardReservoir(res_size; weight = 0.1, fb_weight = 0.2)
    return DelayLineBackwardReservoir(res_size, weight, fb_weight)
end

function create_reservoir(reservoir::DelayLineBackwardReservoir,
                          res_size;
                          matrix_type = Matrix{Float64})
    reservoir_matrix = zeros(res_size, res_size)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = reservoir.weight
        reservoir_matrix[i, i + 1] = reservoir.fb_weight
    end

    return Adapt.adapt(matrix_type, reservoir_matrix)
end

#from "minimum complexity echo state network" Rodan
# Simple cycle reservoir
struct SimpleCycleReservoir{T} <: AbstractReservoir
    res_size::Int
    weight::T
end

"""
    SimpleCycleReservoir(res_size, weight)
    SimpleCycleReservoir(res_size; weight=0.1)

Returns a Simple Cycle Reservoir constructor to build a reservoir matrix as
described in [^Rodan2010]. The `weight` can be passed as an argument or a keyword argument, and it determines the
absolute value of all the connections in the reservoir.

# Arguments
- `res_size::Int`: The size of the reservoir.
- `weight::T`: The weight determines the absolute value of connections in the reservoir.

# Returns
A `SimpleCycleReservoir` object.

# References
[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function SimpleCycleReservoir(res_size; weight = 0.1)
    return SimpleCycleReservoir(res_size, weight)
end

function create_reservoir(reservoir::SimpleCycleReservoir,
                          res_size;
                          matrix_type = Matrix{Float64})
    reservoir_matrix = zeros(Float64, res_size, res_size)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = reservoir.weight
    end

    reservoir_matrix[1, res_size] = reservoir.weight
    return Adapt.adapt(matrix_type, reservoir_matrix)
end

#from "simple deterministically constructed cycle reservoirs with regular jumps" by Rodan and Tino
# Cycle Reservoir with Jumps
struct CycleJumpsReservoir{T} <: AbstractReservoir
    res_size::Int
    cycle_weight::T
    jump_weight::T
    jump_size::Int
end

"""
    CycleJumpsReservoir(res_size; cycle_weight=0.1, jump_weight=0.1, jump_size=3)
    CycleJumpsReservoir(res_size, cycle_weight, jump_weight, jump_size)

Return a Cycle Reservoir with Jumps constructor to create a reservoir matrix as described
in [^Rodan2012]. The `cycle_weight`, `jump_weight`, and `jump_size` can be passed as arguments or keyword arguments, and they
determine the absolute values of connections in the reservoir. The `jump_size` determines the jumps between `jump_weight`s.

# Arguments
- `res_size::Int`: The size of the reservoir.
- `cycle_weight::T`: The weight of cycle connections.
- `jump_weight::T`: The weight of jump connections.
- `jump_size::Int`: The number of steps between jump connections.

# Returns
A `CycleJumpsReservoir` object.

# References
[^Rodan2012]: Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs
with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""
function CycleJumpsReservoir(res_size; cycle_weight = 0.1, jump_weight = 0.1, jump_size = 3)
    return CycleJumpsReservoir(res_size, cycle_weight, jump_weight, jump_size)
end

function create_reservoir(reservoir::CycleJumpsReservoir,
                          res_size;
                          matrix_type = Matrix{Float64})
    reservoir_matrix = zeros(res_size, res_size)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = reservoir.cycle_weight
    end

    reservoir_matrix[1, res_size] = reservoir.cycle_weight

    for i in 1:(reservoir.jump_size):(res_size - reservoir.jump_size)
        tmp = (i + reservoir.jump_size) % res_size
        if tmp == 0
            tmp = res_size
        end
        reservoir_matrix[i, tmp] = reservoir.jump_weight
        reservoir_matrix[tmp, i] = reservoir.jump_weight
    end

    return Adapt.adapt(matrix_type, reservoir_matrix)
end

"""
    NullReservoir()

Return a constructor for a matrix of zeros with dimensions `res_size x res_size`.

# Arguments
- None

# Returns
A `NullReservoir` object.

# References
- None
"""
struct NullReservoir <: AbstractReservoir end

function create_reservoir(reservoir::NullReservoir,
                          res_size;
                          matrix_type = Matrix{Float64})
    return Adapt.adapt(matrix_type, zeros(res_size, res_size))
end
