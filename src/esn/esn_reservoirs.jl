"""
    rand_sparse(rng::AbstractRNG, ::Type{T}, dims::Integer...; radius=1.0, sparsity=0.1)

Create and return a random sparse reservoir matrix for use in Echo State Networks (ESNs). The matrix will be of size specified by `dims`, with specified `sparsity` and scaled spectral radius according to `radius`.

# Arguments
- `rng`: An instance of `AbstractRNG` for random number generation.
- `T`: The data type for the elements of the matrix.
- `dims`: Dimensions of the reservoir matrix.
- `radius`: The desired spectral radius of the reservoir. Defaults to 1.0.
- `sparsity`: The sparsity level of the reservoir matrix, controlling the fraction of zero elements. Defaults to 0.1.

# Returns
A matrix representing the random sparse reservoir.

# References
This type of reservoir initialization is commonly used in ESNs for capturing temporal dependencies in data.
"""
function rand_sparse(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        radius = T(1.0),
        sparsity = T(0.1)) where {T <: Number}
    reservoir_matrix = Matrix{T}(sprand(rng, dims..., sparsity))
    reservoir_matrix = T(2.0) .* (reservoir_matrix .- T(0.5))
    replace!(reservoir_matrix, T(-1.0) => T(0.0))
    rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= radius / rho_w
    if Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix)
        error("Sparsity too low for size of the matrix. Increase res_size or increase sparsity")
    end
    return reservoir_matrix
end

"""
    delay_line(rng::AbstractRNG, ::Type{T}, dims::Integer...; weight=0.1) where {T <: Number}

Create and return a delay line reservoir matrix for use in Echo State Networks (ESNs). A delay line reservoir is a deterministic structure where each unit is connected only to its immediate predecessor with a specified weight. This method is particularly useful for tasks that require specific temporal processing.

# Arguments
- `rng`: An instance of `AbstractRNG` for random number generation. This argument is not used in the current implementation but is included for consistency with other initialization functions.
- `T`: The data type for the elements of the matrix.
- `dims`: Dimensions of the reservoir matrix. Typically, this should be a tuple of two equal integers representing a square matrix.
- `weight`: The weight determines the absolute value of all connections in the reservoir. Defaults to 0.1.

# Returns
A delay line reservoir matrix with dimensions specified by `dims`. The matrix is initialized such that each element in the `i+1`th row and `i`th column is set to `weight`, and all other elements are zeros.

# Example
```julia
reservoir = delay_line(Float64, 100, 100; weight=0.2)
```

# References
This type of reservoir initialization is described in:
Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE Transactions on Neural Networks 22.1 (2010): 131-144.
"""
function delay_line(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        weight = T(0.1)) where {T <: Number}
    reservoir_matrix = zeros(T, dims...)
    @assert length(dims) == 2 && dims[1] == dims[2] "The dimensions must define a square matrix (e.g., (100, 100))"

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    return reservoir_matrix
end

"""
    delay_line_backward_reservoir(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight = T(0.1), fb_weight = T(0.2)) where {T <: Number}

Create a delay line backward reservoir with the specified by `dims` and weights. Creates a matrix with backward connections
as described in [^Rodan2010]. The `weight` and `fb_weight` can be passed as either arguments or
keyword arguments, and they determine the absolute values of the connections in the reservoir.

# Arguments
- `rng::AbstractRNG`: Random number generator.
- `T::Type`: Type of the elements in the reservoir matrix.
- `dims::Integer...`: Dimensions of the reservoir matrix.
- `weight::T`: The weight determines the absolute value of forward connections in the reservoir, and is set to 0.1 by default.
- `fb_weight::T`: The `fb_weight` determines the absolute value of backward connections in the reservoir, and is set to 0.2 by default.


# Returns
Reservoir matrix with the dimensions specified by `dims` and weights.

# References
[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function delay_line_backward_reservoir(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        weight = T(0.1), 
        fb_weight = T(0.2)) where {T <: Number}
    reservoir_matrix = zeros(res_size, res_size)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = weight
        reservoir_matrix[i, i + 1] = fb_weight
    end

    return reservoir_matrix
end


"""
    cycle_jumps_reservoir(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight = T(0.1), jump_weight = T(0.1), jump_size = 3) where {T <: Number}

Create a cycle jumps reservoir with the specified dimensions, cycle weight, jump weight, and jump size.

# Arguments
- `rng::AbstractRNG`: Random number generator.
- `T::Type`: Type of the elements in the reservoir matrix.
- `dims::Integer...`: Dimensions of the reservoir matrix.
- `cycle_weight::T = T(0.1)`:  The weight of cycle connections.
- `jump_weight::T = T(0.1)`: The weight of jump connections.
- `jump_size::Int = 3`:  The number of steps between jump connections.

# Returns
Reservoir matrix with the specified dimensions, cycle weight, jump weight, and jump size.

# References
[^Rodan2012]: Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs
with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""
function cycle_jumps_reservoir(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        cycle_weight = T(0.1), 
        jump_weight = T(0.1), 
        jump_size = T(3)) where {T <: Number}

    reservoir_matrix = zeros(T, dims...)

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = cycle_weight
    end

    reservoir_matrix[1, dims[1]] = cycle_weight

    for i in 1:jump_size:(dims[1] - jump_size)
        tmp = (i + jump_size) % dims[1]
        if tmp == 0
            tmp = dims[1]
        end
        reservoir_matrix[i, tmp] = jump_weight
        reservoir_matrix[tmp, i] = jump_weight
    end

    return reservoir_matrix
end


"""
    simple_cycle_reservoir(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight = T(0.1)) where {T <: Number}

Create a simple cycle reservoir with the specified dimensions and weight.

# Arguments
- `rng::AbstractRNG`: Random number generator.
- `T::Type`: Type of the elements in the reservoir matrix.
- `dims::Integer...`: Dimensions of the reservoir matrix.
- `weight::T = T(0.1)`: Weight of the connections in the reservoir matrix.

# Returns
Reservoir matrix with the dimensions specified by `dims` and weights.

# References
[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function simple_cycle_reservoir(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        weight = T(0.1)) where {T <: Number}
    reservoir_matrix = zeros(T, dims...)

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    reservoir_matrix[1, dims[1]] = weight
    return reservoir_matrix
end




"""
    pseudo_svd_reservoir(rng::AbstractRNG, ::Type{T}, dims::Integer...; 
        max_value, sparsity, sorted = true, reverse_sort = false) where {T <: Number}

    Returns an initializer to build a sparse reservoir matrix with the given `sparsity` by using a pseudo-SVD approach as described in [^yang].

# Arguments
- `rng::AbstractRNG`: Random number generator.
- `T::Type`: Type of the elements in the reservoir matrix.
- `dims::Integer...`: Dimensions of the reservoir matrix.
- `max_value`: The maximum absolute value of elements in the matrix.
- `sparsity`: The desired sparsity level of the reservoir matrix.
- `sorted`: A boolean indicating whether to sort the singular values before creating the diagonal matrix. By default, it is set to `true`.
- `reverse_sort`: A boolean indicating whether to reverse the sorted singular values. By default, it is set to `false`.

# Returns
Reservoir matrix with the specified dimensions, max value, and sparsity.

# References
This reservoir initialization method, based on a pseudo-SVD approach, is inspired by the work in [^yang], which focuses on designing polynomial echo state networks for time series prediction.

[^yang]: Yang, Cuili, et al. "_Design of polynomial echo state networks for time series prediction._" Neurocomputing 290 (2018): 148-160.
"""
function pseudo_svd_reservoir(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...; 
        max_value, sparsity;
        sorted = true, 
        reverse_sort = false) where {T <: Number}

    reservoir_matrix = create_diag( dims[1], max_value, sorted = sorted, reverse_sort = reverse_sort)
    tmp_sparsity = get_sparsity(reservoir_matrix,   dims[1])

    while tmp_sparsity <= sparsity
        reservoir_matrix *= create_qmatrix(dims[1], rand(1:dims[1]), rand(1:dims[1]), rand() * 2 - 1)
        tmp_sparsity = get_sparsity(reservoir_matrix, dims[1])
    end

    return reservoir_matrix
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



# from WeightInitializers.jl, TODO @MartinuzziFrancesco consider importing package
function _default_rng()
    @static if VERSION >= v"1.7"
        return Xoshiro(1234)
    else
        return MersenneTwister(1234)
    end
end

__partial_apply(fn, inp) = fn$inp
