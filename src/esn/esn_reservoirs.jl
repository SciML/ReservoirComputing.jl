"""
    rand_sparse([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...;
        radius=1.0, sparsity=0.1, std=1.0)

Create and return a random sparse reservoir matrix.
The matrix will be of size specified by `dims`, with specified `sparsity`
and scaled spectral radius according to `radius`.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.
  - `radius`: The desired spectral radius of the reservoir.
    Defaults to 1.0.
  - `sparsity`: The sparsity level of the reservoir matrix,
    controlling the fraction of zero elements. Defaults to 0.1.

# Examples

```jldoctest
julia> res_matrix = rand_sparse(5, 5; sparsity=0.5)
5×5 Matrix{Float32}:
 0.0        0.0        0.0        0.0      0.0
 0.0        0.794565   0.0        0.26164  0.0
 0.0        0.0       -0.931294   0.0      0.553706
 0.723235  -0.524727   0.0        0.0      0.0
 1.23723    0.0        0.181824  -1.5478   0.465328
```
"""
function rand_sparse(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        radius=T(1.0), sparsity=T(0.1), std=T(1.0)) where {T <: Number}
    lcl_sparsity = T(1) - sparsity #consistency with current implementations
    reservoir_matrix = sparse_init(rng, T, dims...; sparsity=lcl_sparsity, std=std)
    rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= radius / rho_w
    if Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix)
        error("Sparsity too low for size of the matrix. Increase res_size or increase sparsity")
    end
    return reservoir_matrix
end

"""
    delay_line([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...;
        weight=0.1)

Create and return a delay line reservoir matrix [^Rodan2010].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.
  - `weight`: Determines the value of all connections in the reservoir.
    Default is 0.1.

# Examples

```jldoctest
julia> res_matrix = delay_line(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = delay_line(5, 5; weight=1)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
```

[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
    IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function delay_line(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight=T(0.1)) where {T <: Number}
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    @assert length(dims) == 2&&dims[1] == dims[2] "The dimensions
    must define a square matrix (e.g., (100, 100))"

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    return reservoir_matrix
end

"""
    delay_line_backward([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...;
        weight = 0.1, fb_weight = 0.2)

Create a delay line backward reservoir with the specified by `dims` and weights.
Creates a matrix with backward connections as described in [^Rodan2010].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.
  - `weight`: The weight determines the absolute value of
    forward connections in the reservoir. Default is 0.1
  - `fb_weight`: Determines the absolute value of backward connections
    in the reservoir. Default is 0.2

# Examples

```jldoctest
julia> res_matrix = delay_line_backward(5, 5)
5×5 Matrix{Float32}:
 0.0  0.2  0.0  0.0  0.0
 0.1  0.0  0.2  0.0  0.0
 0.0  0.1  0.0  0.2  0.0
 0.0  0.0  0.1  0.0  0.2
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = delay_line_backward(Float16, 5, 5)
5×5 Matrix{Float16}:
 0.0  0.2  0.0  0.0  0.0
 0.1  0.0  0.2  0.0  0.0
 0.0  0.1  0.0  0.2  0.0
 0.0  0.0  0.1  0.0  0.2
 0.0  0.0  0.0  0.1  0.0
```

[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
    IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function delay_line_backward(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight=T(0.1), fb_weight=T(0.2)) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = weight
        reservoir_matrix[i, i + 1] = fb_weight
    end

    return reservoir_matrix
end

"""
    cycle_jumps([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...; 
        cycle_weight = 0.1, jump_weight = 0.1, jump_size = 3)

Create a cycle jumps reservoir with the specified dimensions,
cycle weight, jump weight, and jump size.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.
  - `cycle_weight`:  The weight of cycle connections.
    Default is 0.1.
  - `jump_weight`: The weight of jump connections.
    Default is 0.1.
  - `jump_size`:  The number of steps between jump connections.
    Default is 3.

# Examples

```jldoctest
julia> res_matrix = cycle_jumps(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.1  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = cycle_jumps(5, 5; jump_size=2)
5×5 Matrix{Float32}:
 0.0  0.0  0.1  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.1  0.1  0.0  0.0  0.1
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.1  0.1  0.0
```

[^Rodan2012]: Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs
    with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""
function cycle_jumps(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Number=T(0.1), jump_weight::Number=T(0.1),
        jump_size::Int=3) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = cycle_weight
    end

    reservoir_matrix[1, res_size] = cycle_weight

    for i in 1:jump_size:(res_size - jump_size)
        tmp = (i + jump_size) % res_size
        if tmp == 0
            tmp = res_size
        end
        reservoir_matrix[i, tmp] = jump_weight
        reservoir_matrix[tmp, i] = jump_weight
    end

    return reservoir_matrix
end

"""
    simple_cycle([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...; 
        weight = 0.1)

Create a simple cycle reservoir with the specified dimensions and weight.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.
  - `weight`: Weight of the connections in the reservoir matrix.
    Default is 0.1.

# Examples

```jldoctest
julia> res_matrix = simple_cycle(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = simple_cycle(5, 5; weight=11)
5×5 Matrix{Float32}:
  0.0   0.0   0.0   0.0  11.0
 11.0   0.0   0.0   0.0   0.0
  0.0  11.0   0.0   0.0   0.0
  0.0   0.0  11.0   0.0   0.0
  0.0   0.0   0.0  11.0   0.0
```

[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
    IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function simple_cycle(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight=T(0.1)) where {T <: Number}
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    reservoir_matrix[1, dims[1]] = weight
    return reservoir_matrix
end

"""
    pseudo_svd([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...; 
        max_value=1.0, sparsity=0.1, sorted = true, reverse_sort = false)

Returns an initializer to build a sparse reservoir matrix with the given
`sparsity` by using a pseudo-SVD approach as described in [^yang].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.
  - `max_value`: The maximum absolute value of elements in the matrix.
    Default is 1.0
  - `sparsity`: The desired sparsity level of the reservoir matrix.
    Default is 0.1
  - `sorted`: A boolean indicating whether to sort the singular values before
    creating the diagonal matrix. Default is `true`.
  - `reverse_sort`: A boolean indicating whether to reverse the sorted
    singular values. Default is `false`.

# Examples

```jldoctest
julia> res_matrix = pseudo_svd(5, 5)
5×5 Matrix{Float32}:
 0.306998  0.0       0.0       0.0       0.0
 0.0       0.325977  0.0       0.0       0.0
 0.0       0.0       0.549051  0.0       0.0
 0.0       0.0       0.0       0.726199  0.0
 0.0       0.0       0.0       0.0       1.0
```

[^yang]: Yang, Cuili, et al. "_Design of polynomial echo state networks for time series prediction._" Neurocomputing 290 (2018): 148-160.
"""
function pseudo_svd(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        max_value::Number=T(1.0), sparsity::Number=0.1, sorted::Bool=true,
        reverse_sort::Bool=false) where {T <: Number}
    reservoir_matrix = create_diag(rng, T, dims[1],
        max_value;
        sorted=sorted,
        reverse_sort=reverse_sort)
    tmp_sparsity = get_sparsity(reservoir_matrix, dims[1])

    while tmp_sparsity <= sparsity
        reservoir_matrix *= create_qmatrix(rng, T, dims[1],
            rand_range(rng, T, dims[1]),
            rand_range(rng, T, dims[1]),
            DeviceAgnostic.rand(rng, T) * T(2) - T(1))
        tmp_sparsity = get_sparsity(reservoir_matrix, dims[1])
    end

    return reservoir_matrix
end

#hacky workaround for the moment
function rand_range(rng, T, n::Int)
    return Int(1 + floor(DeviceAgnostic.rand(rng, T) * n))
end

function create_diag(rng::AbstractRNG, ::Type{T}, dim::Number, max_value::Number;
        sorted::Bool=true, reverse_sort::Bool=false) where {T <: Number}
    diagonal_matrix = DeviceAgnostic.zeros(rng, T, dim, dim)
    if sorted == true
        if reverse_sort == true
            diagonal_values = sort(
                DeviceAgnostic.rand(rng, T, dim) .* max_value; rev=true)
            diagonal_values[1] = max_value
        else
            diagonal_values = sort(DeviceAgnostic.rand(rng, T, dim) .* max_value)
            diagonal_values[end] = max_value
        end
    else
        diagonal_values = DeviceAgnostic.rand(rng, T, dim) .* max_value
    end

    for i in 1:dim
        diagonal_matrix[i, i] = diagonal_values[i]
    end

    return diagonal_matrix
end

function create_qmatrix(rng::AbstractRNG, ::Type{T}, dim::Number,
        coord_i::Number, coord_j::Number, theta::Number) where {T <: Number}
    qmatrix = DeviceAgnostic.zeros(rng, T, dim, dim)

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
