"""
    rand_sparse([rng], [T], dims...;
        radius=1.0, sparsity=0.1, std=1.0, return_sparse=false)

Create and return a random sparse reservoir matrix.
The matrix will be of size specified by `dims`, with specified `sparsity`
and scaled spectral radius according to `radius`.

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `radius`: The desired spectral radius of the reservoir.
    Defaults to 1.0.
  - `sparsity`: The sparsity level of the reservoir matrix,
    controlling the fraction of zero elements. Defaults to 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.

## Examples

Changing the sparsity:

```jldoctest randsparse
julia> res_matrix = rand_sparse(5, 5; sparsity = 0.5)
5×5 Matrix{Float32}:
 0.0        0.0        0.0        0.0      0.0
 0.0        0.794565   0.0        0.26164  0.0
 0.0        0.0       -0.931294   0.0      0.553706
 0.723235  -0.524727   0.0        0.0      0.0
 1.23723    0.0        0.181824  -1.5478   0.465328

julia> res_matrix = rand_sparse(5, 5; sparsity = 0.2)
5×5 Matrix{Float32}:
 0.0       0.0        0.0   0.0      0.0
 0.0       0.853184   0.0   0.0      0.0
 0.0       0.0       -1.0   0.0      0.0
 0.776591  0.0        0.0   0.0      0.0
 0.0       0.0        0.0  -1.66199  0.499657

julia> res_matrix = rand_sparse(5, 5; sparsity = 0.8)
5×5 Matrix{Float32}:
 0.0        0.229011   0.625026    -0.660061  -1.39078
 -0.295761   0.32544    0.0          0.107163   0.0
 0.766352   1.44836   -0.381442    -0.435473   0.226788
 0.296224  -0.214919   0.00956791   0.0        0.210393
 0.506746   0.0        0.0744718   -0.633951   0.19059
```

Returning a sparse matrix:

```jldoctest randsparse
julia> using SparseArrays

julia> res_matrix = rand_sparse(5, 5; sparsity = 0.4, return_sparse = true)
5×5 SparseMatrixCSC{Float32, Int64} with 10 stored entries:
  ⋅          ⋅          ⋅          ⋅        ⋅
  ⋅         0.794565    ⋅         0.26164   ⋅
  ⋅          ⋅        -0.931294    ⋅       0.553706
 0.723235  -0.524727    ⋅          ⋅        ⋅
 1.23723     ⋅         0.181824  -1.5478   0.465328
```
"""
function rand_sparse(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        radius::Number = T(1.0), sparsity::Number = T(0.1), std::Number = T(1.0),
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    lcl_sparsity = T(1) - sparsity #consistency with current implementations
    reservoir_matrix = sparse_init(rng, T, dims...; sparsity = lcl_sparsity, std = std)
    reservoir_matrix = scale_radius!(reservoir_matrix, T(radius))
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

"""
    pseudo_svd([rng], [T], dims...;
        max_value=1.0, sparsity=0.1, sorted=true, reverse_sort=false,
        return_sparse=false)

Returns an initializer to build a sparse reservoir matrix with the given
`sparsity` by using a pseudo-SVD approach as described in [Yang2018](@cite).

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `max_value`: The maximum absolute value of elements in the matrix.
    Default is 1.0
  - `sparsity`: The desired sparsity level of the reservoir matrix.
    Default is 0.1
  - `sorted`: A boolean indicating whether to sort the singular values before
    creating the diagonal matrix. Default is `true`.
  - `reverse_sort`: A boolean indicating whether to reverse the sorted
    singular values. Default is `false`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `return_diag`: flag for returning a `Diagonal` matrix. If both `return_diag`
    and `return_sparse` are set to `true` priority is given to `return_diag`.
    Default is `false`.

## Examples

Default call:

```jldoctest psvd
julia> res_matrix = pseudo_svd(5, 5)
5×5 Matrix{Float32}:
 0.306998  0.0       0.0       0.0       0.0
 0.0       0.325977  0.0       0.0       0.0
 0.0       0.0       0.549051  0.0       0.0
 0.0       0.0       0.0       0.726199  0.0
 0.0       0.0       0.0       0.0       1.0
```

With reversed sorting:

```jldoctest psvd
julia> pseudo_svd(5, 5; reverse_sort = true)
5×5 Matrix{Float32}:
 1.0  0.0       0.0       0.0       0.0
 0.0  0.726199  0.0       0.0       0.0
 0.0  0.0       0.549051  0.0       0.0
 0.0  0.0       0.0       0.325977  0.0
 0.0  0.0       0.0       0.0       0.306998
```

With no sorting

```jldoctest psvd
julia> pseudo_svd(5, 5; sorted = false)
5×5 Matrix{Float32}:
 0.726199  0.0       0.0       0.0       0.0
 0.0       0.325977  0.0       0.0       0.0
 0.0       0.0       0.306998  0.0       0.0
 0.0       0.0       0.0       0.549051  0.0
 0.0       0.0       0.0       0.0       0.788919
```

Returning as a `Diagonal` or a `sparse` matrix:

```jldoctest psvd
julia> pseudo_svd(5, 5; return_diag = true)
5×5 LinearAlgebra.Diagonal{Float32, Vector{Float32}}:
 0.306998   ⋅         ⋅         ⋅         ⋅
  ⋅        0.325977   ⋅         ⋅         ⋅
  ⋅         ⋅        0.549051   ⋅         ⋅
  ⋅         ⋅         ⋅        0.726199   ⋅
  ⋅         ⋅         ⋅         ⋅        1.0

julia> using SparseArrays

julia> pseudo_svd(5, 5; return_sparse = true)
5×5 SparseMatrixCSC{Float32, Int64} with 5 stored entries:
 0.306998   ⋅         ⋅         ⋅         ⋅
  ⋅        0.325977   ⋅         ⋅         ⋅
  ⋅         ⋅        0.549051   ⋅         ⋅
  ⋅         ⋅         ⋅        0.726199   ⋅
  ⋅         ⋅         ⋅         ⋅        1.0
```
"""
function pseudo_svd(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        max_value::Number = T(1),
        sparsity::Number = 0.1f0,
        sorted::Bool = true,
        reverse_sort::Bool = false,
        return_sparse::Bool = false,
        return_diag::Bool = false
    ) where {T <: Number}
    @assert !isempty(dims) "expected at least one dimension"
    res_dim = Int(dims[1])

    reservoir_matrix = create_diag(
        rng, T, res_dim, T(max_value);
        sorted = sorted, reverse_sort = reverse_sort
    )

    tmp = get_sparsity(reservoir_matrix, res_dim)
    while tmp <= sparsity
        i = rand_range(rng, res_dim)
        j = rand_range(rng, res_dim)
        θ = DeviceAgnostic.rand(rng, T) * T(2) .- T(1)
        reservoir_matrix = reservoir_matrix * create_qmatrix(rng, T, res_dim, i, j, θ)
        tmp = get_sparsity(reservoir_matrix, res_dim)
    end

    if return_diag
        return Diagonal(diag(reservoir_matrix))
    else
        return return_init_as(Val(return_sparse), reservoir_matrix)
    end
end

rand_range(rng::AbstractRNG, n::Integer) = rand(rng, 1:n)

function get_sparsity(reservoir_matrix::AbstractMatrix, res_dim::Integer)
    num_notzeros = count(!iszero, reservoir_matrix)
    num_zeros = res_dim * res_dim - num_notzeros
    return num_notzeros / num_zeros
end

function create_diag(
        rng::AbstractRNG, ::Type{T}, res_dim::Integer, max_value::Number;
        sorted::Bool = true, reverse_sort::Bool = false
    ) where {T <: Number}
    diag_matrix = DeviceAgnostic.rand(rng, T, Int(res_dim)) .* T(max_value)

    if sorted
        sort!(diag_matrix)
        if reverse_sort
            reverse!(diag_matrix)
            diag_matrix[1] = T(max_value)
        else
            diag_matrix[end] = T(max_value)
        end
    end

    full_diag = DeviceAgnostic.zeros(rng, T, Int(res_dim), Int(res_dim))
    @inbounds for i in 1:res_dim
        full_diag[i, i] = diag_matrix[i]
    end
    return full_diag
end

function create_qmatrix(
        rng::AbstractRNG, ::Type{T}, n::Integer,
        i::Integer, j::Integer, θ::Number
    ) where {T <: Number}
    Q = DeviceAgnostic.zeros(rng, T, Int(n), Int(n))
    @inbounds for k in 1:n
        Q[k, k] = one(T)
    end
    c = cos(T(θ))
    s = sin(T(θ))
    @inbounds begin
        Q[i, i] = c
        Q[j, j] = c
        Q[i, j] = -s
        Q[j, i] = s
    end
    return Q
end

"""
    chaotic_init([rng], [T], dims...;
        extra_edge_probability=T(0.1), radius=one(T),
        return_sparse=false)

Construct a chaotic reservoir matrix using a digital chaotic system [Xie2024](@cite).

The matrix topology is derived from a strongly connected adjacency
matrix based on a digital chaotic system operating at finite precision.
If the requested matrix order does not exactly match a valid order the
closest valid order is used.

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `extra_edge_probability`: Probability of adding extra random edges in
    the adjacency matrix to enhance connectivity. Default is 0.1.
  - `radius`: The target spectral radius for the
    reservoir matrix. Default is one.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.

## Examples

```jldoctest
julia> res_matrix = chaotic_init(8, 8)
┌ Warning:
│
│     Adjusting reservoir matrix order:
│         from 8 (requested) to 4
│     based on computed bit precision = 1.
│
└ @ ReservoirComputing ~/.julia/dev/ReservoirComputing/src/esn/esn_inits.jl:805
4×4 SparseArrays.SparseMatrixCSC{Float32, Int64} with 6 stored entries:
   ⋅        -0.600945   ⋅          ⋅
   ⋅          ⋅        0.132667   2.21354
   ⋅        -2.60383    ⋅        -2.90391
 -0.578156    ⋅         ⋅          ⋅
```
"""
function chaotic_init(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        extra_edge_probability::AbstractFloat = T(0.1f0), radius::AbstractFloat = one(T),
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    requested_order = first(dims)
    if length(dims) > 1 && dims[2] != requested_order
        @warn """\n
            Using dims[1] = $requested_order for the chaotic reservoir matrix order.\n
        """
    end
    d_estimate = log2(requested_order) / 2
    d_floor = max(floor(Int, d_estimate), 1)
    d_ceil = ceil(Int, d_estimate)
    candidate_order_floor = 2^(2 * d_floor)
    candidate_order_ceil = 2^(2 * d_ceil)
    chosen_bit_precision = abs(candidate_order_floor - requested_order) <=
        abs(candidate_order_ceil - requested_order) ? d_floor : d_ceil
    actual_matrix_order = 2^(2 * chosen_bit_precision)
    if actual_matrix_order != requested_order
        @warn """\n
            Adjusting reservoir matrix order:
                from $requested_order (requested) to $actual_matrix_order
            based on computed bit precision = $chosen_bit_precision. \n
        """
    end

    random_weight_matrix = T(2) * rand(rng, T, actual_matrix_order, actual_matrix_order) .-
        T(1)
    adjacency_matrix = digital_chaotic_adjacency(
        rng, chosen_bit_precision; extra_edge_probability = extra_edge_probability
    )
    reservoir_matrix = random_weight_matrix .* adjacency_matrix
    current_spectral_radius = maximum(abs, eigvals(reservoir_matrix))
    if current_spectral_radius != 0
        reservoir_matrix .*= radius / current_spectral_radius
    end

    return return_init_as(Val(return_sparse), reservoir_matrix)
end

function digital_chaotic_adjacency(
        rng::AbstractRNG, bit_precision::Integer;
        extra_edge_probability::AbstractFloat = 0.1
    )
    matrix_order = 2^(2 * bit_precision)
    adjacency_matrix = DeviceAgnostic.zeros(rng, Int, matrix_order, matrix_order)
    for row_index in 1:(matrix_order - 1)
        adjacency_matrix[row_index, row_index + 1] = 1
    end
    adjacency_matrix[matrix_order, 1] = 1
    for row_index in 1:matrix_order, column_index in 1:matrix_order

        if row_index != column_index && rand(rng) < extra_edge_probability
            adjacency_matrix[row_index, column_index] = 1
        end
    end

    return adjacency_matrix
end

"""
    low_connectivity([rng], [T], dims...;
        connected=false, in_degree = 1, radius = 1.0,
        cut_cycle = false, radius=nothing, return_sparse = false)

Construct an internal reservoir connectivity matrix with low connectivity.

This function creates a reservoir matrix with the specified in-degree
for each node [Griffith2019](@cite). When `in_degree` is 1, the function can enforce
a fully connected cycle if `connected` is `true`;
otherwise, it generates a random connectivity pattern.

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword Arguments

  - `connected`: For `in_degree == 1`, if `true` a connected cycle is enforced.
    Default is `false`.
  - `in_degree`: The number of incoming connections per node.
    Must not exceed the number of nodes. Default is 1.
  - `radius`: The desired spectral radius of the reservoir.
    Defaults to 1.0.
  - `cut_cycle`: If `true`, removes one edge from the cycle to cut it.
    Default is `false`.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.

## Examples

```jldoctest lowcon
julia> low_connectivity(10, 10)
10×10 Matrix{Float32}:
 0.0        0.0       0.0       …  0.0      0.0   0.2207
 0.0        0.0       0.0          0.0      0.0   0.564821
 0.318999   0.0       0.0          0.0      0.0   0.0
 0.670023   0.0       0.0          0.0      0.0   0.0
 0.0        0.0       0.0          1.79705  0.0   0.0
 0.0       -1.95711   0.0       …  0.0      0.0   0.0
 0.0        0.0       0.0          0.0      0.0   0.0
 0.0        0.0       0.0          0.0      0.0   0.0
 0.0        0.0      -0.650657     0.0      0.0   0.0
 0.0        0.0       0.0          0.0      0.0  -1.0
```
"""
function low_connectivity(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        return_sparse::Bool = false, connected::Bool = false,
        in_degree::Integer = 1, radius::Union{AbstractFloat, Nothing} = nothing,
        kwargs...
    ) where {T <: Number}
    check_res_size(dims...)
    res_size = dims[1]
    if in_degree > res_size
        error(
            """
                In-degree k (got k=$(in_degree)) cannot exceed number of nodes N=$(res_size)
            """
        )
    end
    if in_degree == 1
        reservoir_matrix = build_cycle(
            Val(connected), rng, T, res_size; in_degree = in_degree, kwargs...
        )
    else
        reservoir_matrix = build_cycle(
            Val(false), rng, T, res_size; in_degree = in_degree, kwargs...
        )
    end
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

function build_cycle(
        ::Val{false}, rng::AbstractRNG, ::Type{T}, res_size::Int;
        in_degree::Integer = 1, radius::Number = T(1.0f0),
        cut_cycle::Bool = false
    ) where {T <: Number}
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, res_size, res_size)
    for idx in 1:res_size
        selected = randperm(rng, res_size)[1:in_degree]
        for jdx in selected
            reservoir_matrix[idx, jdx] = T(randn(rng))
        end
    end
    reservoir_matrix = scale_radius!(reservoir_matrix, T(radius))
    return reservoir_matrix
end

function build_cycle(
        ::Val{true}, rng::AbstractRNG, ::Type{T}, res_size::Int;
        in_degree::Integer = 1, radius::Number = T(1.0f0), cut_cycle::Bool = false
    ) where {
        T <:
        Number,
    }
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, res_size, res_size)
    perm = randperm(rng, res_size)
    for idx in 1:(res_size - 1)
        reservoir_matrix[perm[idx], perm[idx + 1]] = T(randn(rng))
    end
    reservoir_matrix[perm[res_size], perm[1]] = T(randn(rng))
    reservoir_matrix = scale_radius!(reservoir_matrix, T(radius))
    if cut_cycle
        cut_cycle_edge!(reservoir_matrix, rng)
    end
    return reservoir_matrix
end

function cut_cycle_edge!(
        reservoir_matrix::AbstractMatrix{T}, rng::AbstractRNG
    ) where {T <: Number}
    res_size = size(reservoir_matrix, 1)
    row = rand(rng, 1:res_size)
    for idx in 1:res_size
        if reservoir_matrix[row, idx] != zero(T)
            reservoir_matrix[row, idx] = zero(T)
            break
        end
    end
    return reservoir_matrix
end

@doc raw"""
    delay_line([rng], [T], dims...;
        delay_weight=0.1, delay_shift=1,
        return_sparse=false, radius=nothing, kwargs...)

Create and return a delay line reservoir matrix [Rodan2011](@cite).

```math
W_{i,j} =
\begin{cases}
    r, & \text{if } i = j + 1, j \in [1, D_{\mathrm{res}} - 1], \\[6pt]
    0, & \text{otherwise.}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `delay_weight`: Determines the value of all connections in the reservoir.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1.
  - `delay_shift`: delay line shift. Default is 1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
    If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
    `weight` can be positive with a probability set by `positive_prob`. If set to
    `:irrational_sample!` the `weight` is negative if the decimal number of the
    irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
    assigned a negative sign after the chosen `strides`. `strides` can be a single
    number or an array. Default is `:no_sample`.
  - `positive_prob`: probability of the `weight` being positive when `sampling_type` is
    set to `:bernoulli_sample!`. Default is 0.5.
  - `irrational`: Irrational number whose decimals decide the sign of `weight`.
    Default is `pi`.
  - `start`: Which place after the decimal point the counting starts for the `irrational`
    sign counting. Default is 1.
  - `strides`: number of strides for assigning negative value to a weight. It can be an
    integer or an array. Default is 2.

## Examples

Default call:

```jldoctest delayline
julia> res_matrix = delay_line(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0
```

Changing weights:

```jldoctest delayline
julia> res_matrix = delay_line(5, 5; delay_weight = 1)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
```

Changing weights to a custom array:

```jldoctest delayline
julia> res_matrix = delay_line(5, 5; delay_weight = rand(Float32, 4))
5×5 Matrix{Float32}:
 0.0       0.0       0.0      0.0        0.0
 0.398408  0.0       0.0      0.0        0.0
 0.0       0.624473  0.0      0.0        0.0
 0.0       0.0       0.66302  0.0        0.0
 0.0       0.0       0.0      0.0780818  0.0
```

Changing sign of the weights with different samplings:

```jldoctest delayline
julia> res_matrix = delay_line(5, 5; sampling_type=:irrational_sample!)
5×5 Matrix{Float32}:
 0.0  0.0   0.0   0.0  0.0
 -0.1  0.0   0.0   0.0  0.0
 0.0  0.1   0.0   0.0  0.0
 0.0  0.0  -0.1   0.0  0.0
 0.0  0.0   0.0  -0.1  0.0

julia> res_matrix = delay_line(5, 5; sampling_type=:bernoulli_sample!)
5×5 Matrix{Float32}:
 0.0   0.0  0.0   0.0  0.0
 0.1   0.0  0.0   0.0  0.0
 0.0  -0.1  0.0   0.0  0.0
 0.0   0.0  0.1   0.0  0.0
 0.0   0.0  0.0  -0.1  0.0
```

Shifting the delay line:

```jldoctest delayline
julia> res_matrix = delay_line(5, 5; delay_shift = 3)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
```

Returning as sparse:

```jldoctest delayline
julia> using SparseArrays

julia> res_matrix = delay_line(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 4 stored entries:
  ⋅    ⋅    ⋅    ⋅    ⋅
 0.1   ⋅    ⋅    ⋅    ⋅
  ⋅   0.1   ⋅    ⋅    ⋅
  ⋅    ⋅   0.1   ⋅    ⋅
  ⋅    ⋅    ⋅   0.1   ⋅
```
"""
function delay_line(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        delay_weight::Union{Number, AbstractVector} = T(0.1f0), delay_shift::Integer = 1,
        return_sparse::Bool = false, radius::Union{AbstractFloat, Nothing} = nothing,
        kwargs...
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    delay_line!(rng, reservoir_matrix, T.(delay_weight), delay_shift; kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    delayline_backward([rng], [T], dims...;
        delay_weight=0.1, fb_weight=0.1,
        delay_shift=1, fb_shift=1, return_sparse=false,
        radius=nothing, delay_kwargs=(), fb_kwargs=())

Create a delay line backward reservoir with the specified by `dims` and weights.
Creates a matrix with backward connections as described in [Rodan2011](@cite).

```math
W_{i,j} =
\begin{cases}
    r, & \text{if } i = j + 1,\;\; j \in [1, D_{\mathrm{res}} - 1], \\[4pt]
    b, & \text{if } j = i + 1,\;\; i \in [1, D_{\mathrm{res}} - 1], \\[6pt]
    0, & \text{otherwise.}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `delay_weight`: The weight determines the absolute value of
    forward connections in the reservoir.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1.
  - `fb_weight`: Determines the absolute value of backward connections
    in the reservoir.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1.
  - `fb_shift`: How far the backward connection will be from the diagonal.
    Default is 1.
  - `delay_shift`: delay line shift relative to the diagonal. Default is 1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `delay_kwargs` and `fb_kwargs`: named tuples that control the kwargs for the
    delay line weight and feedback weights respectively. The kwargs are as follows:

      + `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
        If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
        `weight` can be positive with a probability set by `positive_prob`. If set to
        `:irrational_sample!` the `weight` is negative if the decimal number of the
        irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
        assigned a negative sign after the chosen `strides`. `strides` can be a single
        number or an array. Default is `:no_sample`.
      + `positive_prob`: probability of the `weight` being positive when `sampling_type` is
        set to `:bernoulli_sample!`. Default is 0.5.
      + `irrational`: Irrational number whose decimals decide the sign of `weight`.
        Default is `pi`.
      + `start`: Which place after the decimal point the counting starts for the `irrational`
        sign counting. Default is 1.
      + `strides`: number of strides for assigning negative value to a weight. It can be an
        integer or an array. Default is 2.

## Examples

Default call:

```jldoctest dlbackward
julia> res_matrix = delayline_backward(5, 5)
5×5 Matrix{Float32}:
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1
 0.0  0.0  0.0  0.1  0.0
```

Changing weights:

```jldoctest dlbackward
julia> res_matrix = delayline_backward(5, 5; delay_weight = 0.99, fb_weight=-1.0)
5×5 Matrix{Float32}:
 0.0   -1.0    0.0    0.0    0.0
 0.99   0.0   -1.0    0.0    0.0
 0.0    0.99   0.0   -1.0    0.0
 0.0    0.0    0.99   0.0   -1.0
 0.0    0.0    0.0    0.99   0.0
```

Changing weights to custom arrays:

```jldoctest dlbackward
julia> res_matrix = delayline_backward(5, 5; delay_weight = rand(4), fb_weight=.-rand(4))
5×5 Matrix{Float32}:
 0.0       -0.294809   0.0        0.0        0.0
 0.736006   0.0       -0.449479   0.0        0.0
 0.0        0.10892    0.0       -0.60118    0.0
 0.0        0.0        0.482435   0.0       -0.673392
 0.0        0.0        0.0        0.177982   0.0
```

Changing sign of the weights with different samplings:

```jldoctest dlbackward
julia> res_matrix = delayline_backward(5, 5; delay_kwargs=(;sampling_type=:irrational_sample!))
5×5 Matrix{Float32}:
  0.0  0.1   0.0   0.0  0.0
 -0.1  0.0   0.1   0.0  0.0
  0.0  0.1   0.0   0.1  0.0
  0.0  0.0  -0.1   0.0  0.1
  0.0  0.0   0.0  -0.1  0.0

julia> res_matrix = delayline_backward(5, 5; fb_kwargs=(;sampling_type=:bernoulli_sample!))
5×5 Matrix{Float32}:
 0.0  0.1   0.0  0.0   0.0
 0.1  0.0  -0.1  0.0   0.0
 0.0  0.1   0.0  0.1   0.0
 0.0  0.0   0.1  0.0  -0.1
 0.0  0.0   0.0  0.1   0.0
```

Shifting:

```jldoctest dlbackward
julia> res_matrix = delayline_backward(5, 5; delay_shift=3, fb_shift=2)
5×5 Matrix{Float32}:
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0
 0.0  0.0  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
```

Returning as sparse:

```jldoctest dlbackward
julia> using SparseArrays

julia> res_matrix = delayline_backward(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 8 stored entries:
  ⋅   0.1   ⋅    ⋅    ⋅
 0.1   ⋅   0.1   ⋅    ⋅
  ⋅   0.1   ⋅   0.1   ⋅
  ⋅    ⋅   0.1   ⋅   0.1
  ⋅    ⋅    ⋅   0.1   ⋅
```
"""
function delayline_backward(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        delay_weight::Union{Number, AbstractVector} = T(0.1f0),
        fb_weight::Union{Number, AbstractVector} = T(0.1f0), delay_shift::Integer = 1,
        fb_shift::Integer = 1, return_sparse::Bool = false,
        radius::Union{AbstractFloat, Nothing} = nothing,
        delay_kwargs::NamedTuple = NamedTuple(),
        fb_kwargs::NamedTuple = NamedTuple()
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    delay_line!(rng, reservoir_matrix, T.(delay_weight), delay_shift; delay_kwargs...)
    backward_connection!(rng, reservoir_matrix, T.(fb_weight), fb_shift; fb_kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    cycle_jumps([rng], [T], dims...;
        cycle_weight=0.1, jump_weight=0.1, jump_size=3, return_sparse=false,
        radius=nothing, cycle_kwargs=(), jump_kwargs=())

Create a cycle reservoir with jumps [Rodan2012](@cite).

```math
W_{i,j} =
\begin{cases}
    r,   & \text{if } i = j + 1,\;\; j \in [1, D_{\mathrm{res}} - 1], \\[4pt]
    r,   & \text{if } i = 1,\;\; j = D_{\mathrm{res}}, \\[8pt]
    r_j, & \text{if } i = j + \ell, \\[4pt]
    r_j, & \text{if } j = i + \ell, \\[4pt]
    r_j, & \text{if } (i,j) = (1+\ell, 1), \\[4pt]
    r_j, & \text{if } (i,j) = (1,\, D_{\mathrm{res}}+1-\ell), \\[8pt]
    0, & \text{otherwise.}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `cycle_weight`:  The weight of cycle connections.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `jump_weight`: The weight of jump connections.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the jumps
    you want to populate.
    Default is 0.1.
  - `jump_size`:  The number of steps between jump connections.
    Default is 3.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `cycle_kwargs` and `jump_kwargs`: named tuples that control the kwargs for the
    cycle and jump weights respectively. The kwargs are as follows:

      + `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
        If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
        `weight` can be positive with a probability set by `positive_prob`. If set to
        `:irrational_sample!` the `weight` is negative if the decimal number of the
        irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
        assigned a negative sign after the chosen `strides`. `strides` can be a single
        number or an array. Default is `:no_sample`.
      + `positive_prob`: probability of the `weight` being positive when `sampling_type` is
        set to `:bernoulli_sample!`. Default is 0.5.
      + `irrational`: Irrational number whose decimals decide the sign of `weight`.
        Default is `pi`.
      + `start`: Which place after the decimal point the counting starts for the `irrational`
        sign counting. Default is 1.
      + `strides`: number of strides for assigning negative value to a weight. It can be an
        integer or an array. Default is 2.

## Examples

Default call:

```jldoctest cyclejumps
julia> res_matrix = cycle_jumps(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.1  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0
```

Changing weights:

```jldoctest cyclejumps
julia> res_matrix = cycle_jumps(5, 5; jump_weight = 2, cycle_weight=-1)
5×5 Matrix{Float32}:
 0.0   0.0   0.0   2.0  -1.0
-1.0   0.0   0.0   0.0   0.0
 0.0  -1.0   0.0   0.0   0.0
 2.0   0.0  -1.0   0.0   0.0
 0.0   0.0   0.0  -1.0   0.0
```

Changing weights to custom arrays:

```jldoctest cyclejumps
julia> res_matrix = cycle_jumps(5, 5; jump_weight = .-rand(3), cycle_weight=rand(5))
5×5 Matrix{Float32}:
  0.0       0.0       0.0        -0.453905  0.443731
  0.434804  0.0       0.0         0.0       0.0
  0.0       0.520551  0.0         0.0       0.0
 -0.453905  0.0       0.0665751   0.0       0.0
  0.0       0.0       0.0         0.57811   0.0
```

Changing sign of the weights with different samplings:

```jldoctest cyclejumps
julia> res_matrix = cycle_jumps(5, 5; cycle_kwargs = (;sampling_type=:bernoulli_sample!))
5×5 Matrix{Float32}:
 0.0   0.0  0.0   0.1  0.1
 0.1   0.0  0.0   0.0  0.0
 0.0  -0.1  0.0   0.0  0.0
 0.1   0.0  0.1   0.0  0.0
 0.0   0.0  0.0  -0.1  0.0

julia> res_matrix = cycle_jumps(5, 5; jump_kwargs = (;sampling_type=:irrational_sample!))
5×5 Matrix{Float32}:
  0.0  0.0  0.0  -0.1  0.1
  0.1  0.0  0.0   0.0  0.0
  0.0  0.1  0.0   0.0  0.0
 -0.1  0.0  0.1   0.0  0.0
  0.0  0.0  0.0   0.1  0.0
```

Changing cycle jumps length:

```jldoctest cyclejumps
julia> res_matrix = cycle_jumps(5, 5; jump_size = 2)
5×5 Matrix{Float32}:
 0.0  0.0  0.1  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.1  0.1  0.0  0.0  0.1
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.1  0.1  0.0

julia> res_matrix = cycle_jumps(5, 5; jump_size = 4)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.1  0.0  0.0  0.1  0.0
```

Return as a sparse matrix:

```jldoctest cyclejumps
julia> using SparseArrays

julia> res_matrix = cycle_jumps(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 7 stored entries:
  ⋅    ⋅    ⋅   0.1  0.1
 0.1   ⋅    ⋅    ⋅    ⋅
  ⋅   0.1   ⋅    ⋅    ⋅
 0.1   ⋅   0.1   ⋅    ⋅
  ⋅    ⋅    ⋅   0.1   ⋅
```

"""
function cycle_jumps(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        jump_weight::Union{Number, AbstractVector} = T(0.1f0),
        jump_size::Integer = 3, return_sparse::Bool = false,
        radius::Union{AbstractFloat, Nothing} = nothing,
        cycle_kwargs::NamedTuple = NamedTuple(),
        jump_kwargs::NamedTuple = NamedTuple()
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    res_size = first(dims)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    simple_cycle!(rng, reservoir_matrix, T.(cycle_weight); cycle_kwargs...)
    add_jumps!(rng, reservoir_matrix, T.(jump_weight), jump_size; jump_kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    simple_cycle([rng], [T], dims...;
        cycle_weight=0.1, return_sparse=false,
        radius=nothing, kwargs...)

Create a simple cycle reservoir [Rodan2011](@cite).

```math
W_{i,j} =
\begin{cases}
    r, & \text{if } i = j + 1,\;\; j \in [1, D_{\mathrm{res}} - 1], \\[4pt]
    r, & \text{if } i = 1,\;\; j = D_{\mathrm{res}}, \\[6pt]
    0, & \text{otherwise.}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `cycle_weight`: Weight of the connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `sampling_type`: Sampling that decides the distribution of `cycle_weight` negative numbers.
    If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
    `cycle_weight` can be positive with a probability set by `positive_prob`. If set to
    `:irrational_sample!` the `cycle_weight` is negative if the decimal number of the
    irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
    assigned a negative sign after the chosen `strides`. `strides` can be a single
    number or an array. Default is `:no_sample`.
  - `positive_prob`: probability of the `cycle_weight` being positive when `sampling_type` is
    set to `:bernoulli_sample!`. Default is 0.5.
  - `irrational`: Irrational number whose decimals decide the sign of `cycle_weight`.
    Default is `pi`.
  - `start`: Which place after the decimal point the counting starts for the `irrational`
    sign counting. Default is 1.
  - `strides`: number of strides for assigning negative value to a weight. It can be an
    integer or an array. Default is 2.

## Examples

Default call:

```jldoctest scycle
julia> res_matrix = simple_cycle(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0
```

Changing weights:

```jldoctest scycle
julia> res_matrix = simple_cycle(5, 5; cycle_weight=0.99)
5×5 Matrix{Float32}:
 0.0   0.0   0.0   0.0   0.99
 0.99  0.0   0.0   0.0   0.0
 0.0   0.99  0.0   0.0   0.0
 0.0   0.0   0.99  0.0   0.0
 0.0   0.0   0.0   0.99  0.0
```

Changing weights to a custom array:

```jldoctest scycle
julia> res_matrix = simple_cycle(5, 5; cycle_weight=rand(5))
5×5 Matrix{Float32}:
 0.0       0.0        0.0       0.0       0.471823
 0.534782  0.0        0.0       0.0       0.0
 0.0       0.0764598  0.0       0.0       0.0
 0.0       0.0        0.507883  0.0       0.0
 0.0       0.0        0.0       0.546656  0.0
```

Changing sign of the weights with different samplings:

```jldoctest scycle
julia> res_matrix = simple_cycle(5, 5; sampling_type=:irrational_sample!)
5×5 Matrix{Float32}:
  0.0  0.0   0.0   0.0  -0.1
 -0.1  0.0   0.0   0.0   0.0
  0.0  0.1   0.0   0.0   0.0
  0.0  0.0  -0.1   0.0   0.0
  0.0  0.0   0.0  -0.1   0.0

julia> res_matrix = simple_cycle(5, 5; sampling_type=:bernoulli_sample!)
5×5 Matrix{Float32}:
 0.0   0.0  0.0   0.0  0.1
 0.1   0.0  0.0   0.0  0.0
 0.0  -0.1  0.0   0.0  0.0
 0.0   0.0  0.1   0.0  0.0
 0.0   0.0  0.0  -0.1  0.0
```

Returning as sparse:

```jldoctest scycle
julia> using SparseArrays

julia> res_matrix = simple_cycle(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 5 stored entries:
  ⋅    ⋅    ⋅    ⋅   0.1
 0.1   ⋅    ⋅    ⋅    ⋅
  ⋅   0.1   ⋅    ⋅    ⋅
  ⋅    ⋅   0.1   ⋅    ⋅
  ⋅    ⋅    ⋅   0.1   ⋅
```
"""
function simple_cycle(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        return_sparse::Bool = false, radius::Union{AbstractFloat, Nothing} = nothing,
        kwargs...
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    simple_cycle!(rng, reservoir_matrix, T.(cycle_weight); kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    double_cycle([rng], [T], dims...;
        cycle_weight=0.1, second_cycle_weight=0.1,
        radius=nothing, return_sparse=false)

Creates a double cycle reservoir [Fu2023](@cite).

```math
W_{i,j} =
\begin{cases}
    r_1, & \text{if } i = j + 1,\;\; j \in [1, D_{\mathrm{res}} - 1], \\[4pt]
    r_1, & \text{if } i = D_{\mathrm{res}},\;\; j = 1, \\[6pt]
    r_2, & \text{if } i = 1,\;\; j = D_{\mathrm{res}}, \\[6pt]
    r_2, & \text{if } j = i + 1,\;\; i \in [1, D_{\mathrm{res}} - 1], \\[4pt]
    0,   & \text{otherwise.}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `cycle_weight`: Weight of the upper cycle connections in the reservoir matrix.
    Default is 0.1.
  - `second_cycle_weight`: Weight of the lower cycle connections in the reservoir matrix.
    Default is 0.1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.

## Examples

Default call:

```jldoctest dcycle
julia> res_matrix = double_cycle(5, 5)
5×5 Matrix{Float32}:
 0.0  0.1  0.0  0.0  0.1
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1
 0.1  0.0  0.0  0.1  0.0
```

Changing weights:

```jldoctest dcycle
julia> res_matrix = double_cycle(5, 5; cycle_weight = -0.1, second_cycle_weight = 0.3)
5×5 Matrix{Float32}:
  0.0   0.3   0.0   0.0  0.3
 -0.1   0.0   0.3   0.0  0.0
  0.0  -0.1   0.0   0.3  0.0
  0.0   0.0  -0.1   0.0  0.3
 -0.1   0.0   0.0  -0.1  0.0
```
"""
function double_cycle(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        second_cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        radius::Union{AbstractFloat, Nothing} = nothing,
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for uidx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - 1)
        reservoir_matrix[uidx + 1, uidx] = T.(cycle_weight)
    end
    for lidx in (first(axes(reservoir_matrix, 1)) + 1):last(axes(reservoir_matrix, 1))
        reservoir_matrix[lidx - 1, lidx] = T.(second_cycle_weight)
    end

    reservoir_matrix[1, dims[1]] = T.(second_cycle_weight)
    reservoir_matrix[dims[1], 1] = T.(cycle_weight)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    true_doublecycle([rng], [T], dims...;
        cycle_weight=0.1, second_cycle_weight=0.1, radius=nothing,
        return_sparse=false, cycle_kwargs=(), second_cycle_kwargs=())

Creates a true double cycle reservoir, ispired by [Fu2023](@cite),
with cycles built on the definition by [Rodan2011](@cite).

```math
W_{i,j} =
\begin{cases}
    r_1, & \text{if } i = j + 1,\;\; j \in [1, D_{\mathrm{res}} - 1], \\[4pt]
    r_1, & \text{if } i = 1,\;\; j = D_{\mathrm{res}}, \\[6pt]
    r_2, & \text{if } j = i + 1,\;\; i \in [1, D_{\mathrm{res}} - 1], \\[4pt]
    r_2, & \text{if } i = D_{\mathrm{res}},\;\; j = 1, \\[6pt]
    0,   & \text{otherwise.}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `cycle_weight`: Weight of the upper cycle connections in the reservoir matrix.
    Default is 0.1.
  - `second_cycle_weight`: Weight of the lower cycle connections in the reservoir matrix.
    Default is 0.1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `cycle_kwargs`, and `second_cycle_kwargs`: named tuples that control the kwargs
    for the weights generation. The kwargs are as follows:

      + `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
        If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
        `weight` can be positive with a probability set by `positive_prob`. If set to
        `:irrational_sample!` the `weight` is negative if the decimal number of the
        irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
        assigned a negative sign after the chosen `strides`. `strides` can be a single
        number or an array. Default is `:no_sample`.
      + `positive_prob`: probability of the `weight` being positive when `sampling_type` is
        set to `:bernoulli_sample!`. Default is 0.5.
      + `irrational`: Irrational number whose decimals decide the sign of `weight`.
        Default is `pi`.
      + `start`: Which place after the decimal point the counting starts for the `irrational`
        sign counting. Default is 1.
      + `strides`: number of strides for assigning negative value to a weight. It can be an
        integer or an array. Default is 2.

## Examples

Default call:

```jldoctest tdcycle
julia> res_matrix = true_doublecycle(5, 5)
5×5 Matrix{Float32}:
 0.0  0.1  0.0  0.0  0.1
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1
 0.1  0.0  0.0  0.1  0.0
```

Changing weights:

```jldoctest tdcycle
julia> res_matrix = true_doublecycle(5, 5; cycle_weight = 0.1, second_cycle_weight = 0.3)
5×5 Matrix{Float32}:
 0.0  0.3  0.0  0.0  0.1
 0.1  0.0  0.3  0.0  0.0
 0.0  0.1  0.0  0.3  0.0
 0.0  0.0  0.1  0.0  0.3
 0.3  0.0  0.0  0.1  0.0
```

Changing weights to custom arrays:

```jldoctest tdcycle
julia> res_matrix = true_doublecycle(5, 5; cycle_weight = rand(5), second_cycle_weight = .-rand(5))
5×5 Matrix{Float32}:
  0.0       -0.647066   0.0        0.0        0.604095
  0.6687     0.0       -0.853307   0.0        0.0
  0.0        0.40399    0.0       -0.565928   0.0
  0.0        0.0        0.960196   0.0       -0.120321
 -0.120321   0.0        0.0        0.874008   0.0
```

Changing sign of the weights with different samplings:

```jldoctest tdcycle
julia> res_matrix = true_doublecycle(5, 5; cycle_kwargs=(;sampling_type=:irrational_sample!))
5×5 Matrix{Float32}:
  0.0  0.1   0.0   0.0  -0.1
 -0.1  0.0   0.1   0.0   0.0
  0.0  0.1   0.0   0.1   0.0
  0.0  0.0  -0.1   0.0   0.1
  0.1  0.0   0.0  -0.1   0.0

julia> res_matrix = true_doublecycle(5, 5; second_cycle_kwargs=(;sampling_type=:bernoulli_sample!))
5×5 Matrix{Float32}:
 0.0  -0.1  0.0   0.0  0.1
 0.1   0.0  0.1   0.0  0.0
 0.0   0.1  0.0  -0.1  0.0
 0.0   0.0  0.1   0.0  0.1
 0.1   0.0  0.0   0.1  0.0
```

Returning as sparse:

```jldoctest tdcycle
julia> res_matrix = true_doublecycle(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 10 stored entries:
  ⋅   0.1   ⋅    ⋅   0.1
 0.1   ⋅   0.1   ⋅    ⋅
  ⋅   0.1   ⋅   0.1   ⋅
  ⋅    ⋅   0.1   ⋅   0.1
 0.1   ⋅    ⋅   0.1   ⋅
```
"""
function true_doublecycle(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        second_cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        return_sparse::Bool = false, radius::Union{AbstractFloat, Nothing} = nothing,
        cycle_kwargs::NamedTuple = NamedTuple(),
        second_cycle_kwargs::NamedTuple = NamedTuple()
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    simple_cycle!(rng, reservoir_matrix, cycle_weight; cycle_kwargs...)
    reverse_simple_cycle!(
        rng, reservoir_matrix, second_cycle_weight; second_cycle_kwargs...
    )
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_cycle([rng], [T], dims...;
        cycle_weight=0.1, selfloop_weight=0.1,
        radius=nothing, return_sparse=false, kwargs...)

Creates a simple cycle reservoir with the
addition of self loops [Elsarraj2019](@cite).

This architecture is referred to as TP1 in the original paper.

```math
W_{i,j} =
\begin{cases}
    ll, & \text{if } i = j \\
    r, & \text{if } j = i - 1 \text{ for } i = 2 \dots N \\
    r, & \text{if } i = 1, j = N \\
    0, & \text{otherwise}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `cycle_weight`: Weight of the cycle connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `selfloop_weight`: Weight of the self loops in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the diagonal
    you want to populate.
    Default is 0.1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `cycle_kwargs` and `jump_kwargs`: named tuples that control the kwargs for the
    cycle and jump weights respectively. The kwargs are as follows:

      + `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
        If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
        `weight` can be positive with a probability set by `positive_prob`. If set to
        `:irrational_sample!` the `weight` is negative if the decimal number of the
        irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
        assigned a negative sign after the chosen `strides`. `strides` can be a single
        number or an array. Default is `:no_sample`.
      + `positive_prob`: probability of the `weight` being positive when `sampling_type` is
        set to `:bernoulli_sample!`. Default is 0.5.
      + `irrational`: Irrational number whose decimals decide the sign of `weight`.
        Default is `pi`.
      + `start`: Which place after the decimal point the counting starts for the `irrational`
        sign counting. Default is 1.
      + `strides`: number of strides for assigning negative value to a weight. It can be an
        integer or an array. Default is 2.

## Examples

Default call:

```jldoctest slcycle
julia> res_matrix = selfloop_cycle(5, 5)
5×5 Matrix{Float32}:
 0.1  0.0  0.0  0.0  0.1
 0.1  0.1  0.0  0.0  0.0
 0.0  0.1  0.1  0.0  0.0
 0.0  0.0  0.1  0.1  0.0
 0.0  0.0  0.0  0.1  0.1
```jldoctest slcycle

Changing weights:

```jldoctest slcycle
julia> res_matrix = selfloop_cycle(5, 5; cycle_weight=-0.2, selfloop_weight=0.5)
5×5 Matrix{Float32}:
  0.5   0.0   0.0   0.0  -0.2
 -0.2   0.5   0.0   0.0   0.0
  0.0  -0.2   0.5   0.0   0.0
  0.0   0.0  -0.2   0.5   0.0
  0.0   0.0   0.0  -0.2   0.5
```

Changing weights to custom arrays:

```jldoctest slcycle
julia> res_matrix = selfloop_cycle(5, 5; cycle_weight=rand(5), selfloop_weight=.-rand(5))
5×5 Matrix{Float32}:
 -0.902546   0.0          0.0        0.0          0.0987988
  0.911585  -0.968998     0.0        0.0          0.0
  0.0        0.00149246  -0.613033   0.0          0.0
  0.0        0.0          0.777804  -0.727024     0.0
  0.0        0.0          0.0        0.00441047  -0.310635
```

Changing sign of the weights with different samplings:

```jldoctest slcycle
julia> res_matrix = selfloop_cycle(5, 5; cycle_kwargs=(;sampling_type=:irrational_sample!))
5×5 Matrix{Float32}:
  0.1  0.0   0.0   0.0  -0.1
 -0.1  0.1   0.0   0.0   0.0
  0.0  0.1   0.1   0.0   0.0
  0.0  0.0  -0.1   0.1   0.0
  0.0  0.0   0.0  -0.1   0.1

julia> res_matrix = selfloop_cycle(5, 5; selfloop_kwargs=(;sampling_type=:bernoulli_sample!))
5×5 Matrix{Float32}:
 0.1   0.0  0.0   0.0  0.1
 0.1  -0.1  0.0   0.0  0.0
 0.0   0.1  0.1   0.0  0.0
 0.0   0.0  0.1  -0.1  0.0
 0.0   0.0  0.0   0.1  0.1
```

Returning as sparse:

```jldoctest slcycle
julia> using SparseArrays

julia> res_matrix = selfloop_cycle(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 10 stored entries:
 0.1   ⋅    ⋅    ⋅   0.1
 0.1  0.1   ⋅    ⋅    ⋅
  ⋅   0.1  0.1   ⋅    ⋅
  ⋅    ⋅   0.1  0.1   ⋅
  ⋅    ⋅    ⋅   0.1  0.1
```
"""
function selfloop_cycle(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0),
        radius::Union{AbstractFloat, Nothing} = nothing,
        return_sparse::Bool = false, selfloop_kwargs::NamedTuple = NamedTuple(),
        cycle_kwargs::NamedTuple = NamedTuple()
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    self_loop!(rng, reservoir_matrix, T.(selfloop_weight); selfloop_kwargs...)
    simple_cycle!(rng, reservoir_matrix, T.(cycle_weight); cycle_kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_backward_cycle([rng], [T], dims...;
        cycle_weight=0.1, selfloop_weight=0.1,
        fb_weight = 0.1, radius=nothing, return_sparse=false)

Creates a cycle reservoir with feedback connections on even neurons and
self loops on odd neurons [Elsarraj2019](@cite).

This architecture is referred to as TP2 in the original paper.

```math
W_{i,j} =
\begin{cases}
    r, & \text{if } j = i - 1 \text{ for } i = 2 \dots N \\
    r, & \text{if } i = 1, j = N \\
    ll, & \text{if } i = j \text{ and } i \text{ is odd} \\
    r, & \text{if } j = i + 1 \text{ and } i \text{ is even}, i \neq N \\
    0, & \text{otherwise}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `cycle_weight`: Weight of the cycle connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `selfloop_weight`: Weight of the self loops in the reservoir matrix.
    Default is 0.1.
  - `fb_weight`: Weight of the self loops in the reservoir matrix.
    Default is 0.1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.

## Examples

```jldoctest
julia> reservoir_matrix = selfloop_backward_cycle(5, 5)
5×5 Matrix{Float32}:
 0.1  0.1  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.1  0.1  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.1

julia> reservoir_matrix = selfloop_backward_cycle(5, 5; self_loop_weight=0.5)
5×5 Matrix{Float32}:
 0.5  0.1  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.5  0.1  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.5
```
"""
function selfloop_backward_cycle(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0),
        fb_weight::Union{Number, AbstractVector} = T(0.1f0),
        radius::Union{AbstractFloat, Nothing} = nothing,
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = simple_cycle(
        rng, T, dims...;
        cycle_weight = T.(cycle_weight), return_sparse = false
    )
    for idx in axes(reservoir_matrix, 1)
        if isodd(idx)
            reservoir_matrix[idx, idx] = T.(selfloop_weight)
        end
    end
    for idx in (first(axes(reservoir_matrix, 1)) + 1):last(axes(reservoir_matrix, 1))
        if iseven(idx)
            reservoir_matrix[idx - 1, idx] = T.(fb_weight)
        end
    end
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_delayline_backward([rng], [T], dims...;
        delay_weight=0.1, selfloop_weight=0.1, fb_weight=0.1,
        fb_shift=2, delya_shift=1, radius=nothing, return_sparse=false,
        fb_kwargs=(), selfloop_kwargs=(), delay_kwargs=())

Creates a reservoir based on a delay line with the addition of self loops and
backward connections shifted by one [Elsarraj2019](@cite).

This architecture is referred to as TP3 in the original paper.

```math
W_{i,j} =
\begin{cases}
    ll, & \text{if } i = j \text{ for } i = 1 \dots N \\
    r, & \text{if } j = i - 1 \text{ for } i = 2 \dots N \\
    r, & \text{if } j = i - 2 \text{ for } i = 3 \dots N \\
    0, & \text{otherwise}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `delay_weight`: Weight of the delay line connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `selfloop_weight`: Weight of the self loops in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the diagonal
    you want to populate.
    Default is 0.1.
  - `fb_weight`: Weight of the feedback in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the diagonal
    you want to populate.
    Default is 0.1.
  - `fb_shift`: How far the backward connection will be from the diagonal.
    Default is 1.
  - `delay_shift`: delay line shift relative to the diagonal. Default is 1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `delay_kwargs`, `selfloop_kwargs`, and `fb_kwargs`: named tuples that control the kwargs
    for the weights generation. The kwargs are as follows:

    + `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
        If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
        `weight` can be positive with a probability set by `positive_prob`. If set to
        `:irrational_sample!` the `weight` is negative if the decimal number of the
        irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
        assigned a negative sign after the chosen `strides`. `strides` can be a single
        number or an array. Default is `:no_sample`.
      + `positive_prob`: probability of the `weight` being positive when `sampling_type` is
        set to `:bernoulli_sample!`. Default is 0.5.
      + `irrational`: Irrational number whose decimals decide the sign of `weight`.
        Default is `pi`.
      + `start`: Which place after the decimal point the counting starts for the `irrational`
        sign counting. Default is 1.
      + `strides`: number of strides for assigning negative value to a weight. It can be an
        integer or an array. Default is 2.

## Examples

Default call:

```jldoctest sldlfb
julia> res_matrix = selfloop_delayline_backward(5, 5)
5×5 Matrix{Float32}:
 0.1  0.0  0.1  0.0  0.0
 0.1  0.1  0.0  0.1  0.0
 0.0  0.1  0.1  0.0  0.1
 0.0  0.0  0.1  0.1  0.0
 0.0  0.0  0.0  0.1  0.1
```

Changing weights:

```jldoctest sldlfb
julia> res_matrix = selfloop_delayline_backward(5, 5; selfloop_weight=0.3, fb_weight=0.99, delay_weight=-0.5)
5×5 Matrix{Float32}:
  0.3   0.0   0.99   0.0   0.0
 -0.5   0.3   0.0    0.99  0.0
  0.0  -0.5   0.3    0.0   0.99
  0.0   0.0  -0.5    0.3   0.0
  0.0   0.0   0.0   -0.5   0.3
```

Changing weights to custom arrays:
```jldoctest sldlfb
julia> res_matrix = selfloop_delayline_backward(5, 5; selfloop_weight=randn(5), fb_weight=rand(5), delay_weight=-rand(5))
5×5 Matrix{Float32}:
 -1.22847    0.0       0.384073   0.0        0.0
 -0.699175   2.63937   0.0        0.345408   0.0
  0.0       -0.5171   -0.452312   0.0        0.0205082
  0.0        0.0      -0.193893   1.45921    0.0
  0.0        0.0       0.0       -0.453015  -1.43402
```

Changing sign of the weights with different samplings:

```jldoctest sldlfb
julia> res_matrix = selfloop_delayline_backward(5, 5; selfloop_kwargs=(;sampling_type=:irrational_sample!))
5×5 Matrix{Float32}:
 -0.1  0.0   0.1   0.0   0.0
  0.1  0.1   0.0   0.1   0.0
  0.0  0.1  -0.1   0.0   0.1
  0.0  0.0   0.1  -0.1   0.0
  0.0  0.0   0.0   0.1  -0.1

julia> res_matrix = selfloop_delayline_backward(5, 5; delay_kwargs=(;sampling_type=:bernoulli_sample!))
5×5 Matrix{Float32}:
 0.1   0.0  0.1   0.0  0.0
 0.1   0.1  0.0   0.1  0.0
 0.0  -0.1  0.1   0.0  0.1
 0.0   0.0  0.1   0.1  0.0
 0.0   0.0  0.0  -0.1  0.1

julia> res_matrix = selfloop_delayline_backward(5, 5; fb_kwargs=(;sampling_type=:regular_sample!))
5×5 Matrix{Float32}:
 0.1  0.0  0.1   0.0  0.0
 0.1  0.1  0.0  -0.1  0.0
 0.0  0.1  0.1   0.0  0.1
 0.0  0.0  0.1   0.1  0.0
 0.0  0.0  0.0   0.1  0.1
```

Shifting the delay and the backward line:

```jldoctest sldlfb
julia> res_matrix = selfloop_delayline_backward(5, 5; delay_shift=3, fb_shift=2)
5×5 Matrix{Float32}:
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1
 0.1  0.0  0.0  0.1  0.0
 0.0  0.1  0.0  0.0  0.1
```

Returning as sparse:

```jldoctest sldlfb
julia> using SparseArrays

julia> res_matrix = selfloop_delayline_backward(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 12 stored entries:
 0.1   ⋅   0.1   ⋅    ⋅
 0.1  0.1   ⋅   0.1   ⋅
  ⋅   0.1  0.1   ⋅   0.1
  ⋅    ⋅   0.1  0.1   ⋅
  ⋅    ⋅    ⋅   0.1  0.1
```
"""
function selfloop_delayline_backward(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        delay_shift::Integer = 1, fb_shift::Integer = 2,
        delay_weight::Union{Number, AbstractVector} = T(0.1f0),
        fb_weight::Union{Number, AbstractVector} = delay_weight,
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0),
        return_sparse::Bool = false, radius::Union{AbstractFloat, Nothing} = nothing,
        delay_kwargs::NamedTuple = NamedTuple(),
        fb_kwargs::NamedTuple = NamedTuple(),
        selfloop_kwargs::NamedTuple = NamedTuple()
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    self_loop!(rng, reservoir_matrix, T.(selfloop_weight); selfloop_kwargs...)
    delay_line!(rng, reservoir_matrix, T.(delay_weight), delay_shift; delay_kwargs...)
    backward_connection!(rng, reservoir_matrix, T.(fb_weight), fb_shift; fb_kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_forwardconnection([rng], [T], dims...;
        delay_weight=0.1, selfloop_weight=0.1,
        radius=nothing, return_sparse=false,
        selfloop_kwargs=(), delay_kwargs=())

Creates a reservoir based on a forward connection of weights between even nodes
with the addition of self loops [Elsarraj2019](@cite).

This architecture is referred to as TP4 in the original paper.

```math
W_{i,j} =
\begin{cases}
    ll, & \text{if } i = j \text{ for } i = 1 \dots N \\
    r, & \text{if } j = i - 2 \text{ for } i = 3 \dots N \\
    0, & \text{otherwise}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `forward_weight`: Weight of the forward connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `selfloop_weight`: Weight of the self loops in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the diagonal
    you want to populate.
    Default is 0.1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `delay_kwargs` and `selfloop_kwargs`: named tuples that control the kwargs for the
    delay line weight and self loop weights respectively. The kwargs are as follows:

    + `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
        If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
        `weight` can be positive with a probability set by `positive_prob`. If set to
        `:irrational_sample!` the `weight` is negative if the decimal number of the
        irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
        assigned a negative sign after the chosen `strides`. `strides` can be a single
        number or an array. Default is `:no_sample`.
      + `positive_prob`: probability of the `weight` being positive when `sampling_type` is
        set to `:bernoulli_sample!`. Default is 0.5.
      + `irrational`: Irrational number whose decimals decide the sign of `weight`.
        Default is `pi`.
      + `start`: Which place after the decimal point the counting starts for the `irrational`
        sign counting. Default is 1.
      + `strides`: number of strides for assigning negative value to a weight. It can be an
        integer or an array. Default is 2.

## Examples

Default call:

```jldoctest slfc
julia> res_matrix = selfloop_forwardconnection(5, 5)
5×5 Matrix{Float32}:
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1
```

Changing weights:

```jldoctest slfc
julia> res_matrix = selfloop_forwardconnection(5, 5; forward_weight=0.5, selfloop_weight=0.99)
5×5 Matrix{Float32}:
 0.99  0.0   0.0   0.0   0.0
 0.0   0.99  0.0   0.0   0.0
 0.5   0.0   0.99  0.0   0.0
 0.0   0.5   0.0   0.99  0.0
 0.0   0.0   0.5   0.0   0.99
```

Changing weights to custom arrays:

```jldoctest slfc
julia> res_matrix = selfloop_forwardconnection(5, 5; forward_weight=rand(5), selfloop_weight=.-rand(5))
5×5 Matrix{Float32}:
 -0.0420509   0.0        0.0        0.0        0.0
  0.0        -0.116113   0.0        0.0        0.0
  0.69173     0.0       -0.513592   0.0        0.0
  0.0         0.522245   0.0       -0.199966   0.0
  0.0         0.0        0.784556   0.0       -0.918653
```

```jldoctest slfc
julia> res_matrix = selfloop_forwardconnection(5, 5; delay_kwargs=(;sampling_type=:irrational_sample!))
5×5 Matrix{Float32}:
  0.1  0.0   0.0  0.0  0.0
  0.0  0.1   0.0  0.0  0.0
 -0.1  0.0   0.1  0.0  0.0
  0.0  0.1   0.0  0.1  0.0
  0.0  0.0  -0.1  0.0  0.1

julia> res_matrix = selfloop_forwardconnection(5, 5; selfloop_kwargs=(;sampling_type=:bernoulli_sample!))
5×5 Matrix{Float32}:
 0.1   0.0  0.0   0.0  0.0
 0.0  -0.1  0.0   0.0  0.0
 0.1   0.0  0.1   0.0  0.0
 0.0   0.1  0.0  -0.1  0.0
 0.0   0.0  0.1   0.0  0.1
```

Returning as sparse:

```jldoctest slfc
julia> using SparseArrays

julia> res_matrix = selfloop_forwardconnection(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 8 stored entries:
 0.1   ⋅    ⋅    ⋅    ⋅
  ⋅   0.1   ⋅    ⋅    ⋅
 0.1   ⋅   0.1   ⋅    ⋅
  ⋅   0.1   ⋅   0.1   ⋅
  ⋅    ⋅   0.1   ⋅   0.1
```
"""
function selfloop_forwardconnection(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        forward_weight::Union{Number, AbstractVector} = T(0.1f0),
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0), shift::Integer = 2,
        return_sparse::Bool = false, radius::Union{AbstractFloat, Nothing} = nothing,
        delay_kwargs::NamedTuple = NamedTuple(),
        selfloop_kwargs::NamedTuple = NamedTuple()
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    self_loop!(rng, reservoir_matrix, T.(selfloop_weight); selfloop_kwargs...)
    delay_line!(rng, reservoir_matrix, T.(forward_weight), shift; delay_kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    forward_connection([rng], [T], dims...;
        forward_weight=0.1, radius=nothing, return_sparse=false,
        kwargs...)

Creates a reservoir based on a forward connection of weights [Elsarraj2019](@cite).

This architecture is referred to as TP5 in the original paper.

```math
W_{i,j} =
\begin{cases}
    r, & \text{if } j = i - 2 \text{ for } i = 3 \dots N \\
    0, & \text{otherwise}
\end{cases}
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `forward_weight`: Weight of the cycle connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `sampling_type`: Sampling that decides the distribution of `forward_weight` negative numbers.
    If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
    `forward_weight` can be positive with a probability set by `positive_prob`. If set to
    `:irrational_sample!` the `forward_weight` is negative if the decimal number of the
    irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
    assigned a negative sign after the chosen `strides`. `strides` can be a single
    number or an array. Default is `:no_sample`.
  - `positive_prob`: probability of the `forward_weight` being positive when `sampling_type` is
    set to `:bernoulli_sample!`. Default is 0.5.
  - `irrational`: Irrational number whose decimals decide the sign of `forward_weight`.
    Default is `pi`.
  - `start`: Which place after the decimal point the counting starts for the `irrational`
    sign counting. Default is 1.
  - `strides`: number of strides for assigning negative value to a weight. It can be an
    integer or an array. Default is 2.

## Examples

Default kwargs:

```jldoctest forcon
julia> reservoir_matrix = forward_connection(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
```

Changing the weights magnitudes to a different unique value:

```jldoctest forcon
julia> forward_connection(5, 5; forward_weight=0.99)
5×5 Matrix{Float32}:
 0.0   0.0   0.0   0.0  0.0
 0.0   0.0   0.0   0.0  0.0
 0.99  0.0   0.0   0.0  0.0
 0.0   0.99  0.0   0.0  0.0
 0.0   0.0   0.99  0.0  0.0
```

Changing the weights signs with different sampling techniques:

```jldoctest forcon
julia> forward_connection(5, 5; sampling_type=:irrational_sample!)
5×5 Matrix{Float32}:
  0.0  0.0   0.0  0.0  0.0
  0.0  0.0   0.0  0.0  0.0
 -0.1  0.0   0.0  0.0  0.0
  0.0  0.1   0.0  0.0  0.0
  0.0  0.0  -0.1  0.0  0.0

julia> forward_connection(5, 5; sampling_type=:irrational_sample!)
5×5 Matrix{Float32}:
  0.0  0.0   0.0  0.0  0.0
  0.0  0.0   0.0  0.0  0.0
  -0.1  0.0   0.0  0.0  0.0
  0.0  0.1   0.0  0.0  0.0
  0.0  0.0  -0.1  0.0  0.0
```

Changing the weights to random numbers. Note that the length of the given array
must be at least as long as the subdiagonal one wants to fill:

```jldoctest forcon
julia> reservoir_matrix = forward_connection(5, 5; forward_weight=rand(Float32, 3))
5×5 Matrix{Float32}:
 0.0       0.0       0.0       0.0  0.0
 0.0       0.0       0.0       0.0  0.0
 0.274221  0.0       0.0       0.0  0.0
 0.0       0.111511  0.0       0.0  0.0
 0.0       0.0       0.618345  0.0  0.0
```

```jldoctest forcon

```

Returning a sparse matrix:

```jldoctest forcon

julia> reservoir_matrix = forward_connection(10, 10; return_sparse=true)
10×10 SparseMatrixCSC{Float32, Int64} with 8 stored entries:
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
 0.1   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
  ⋅   0.1   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅   0.1   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅   0.1   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅   0.1   ⋅    ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   0.1   ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   0.1   ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   0.1   ⋅    ⋅
```


"""
function forward_connection(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        forward_weight::Union{Number, AbstractVector} = T(0.1f0),
        radius::Union{AbstractFloat, Nothing} = nothing, return_sparse::Bool = false,
        kwargs...
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    delay_line!(rng, reservoir_matrix, T.(forward_weight), 2; kwargs...)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    block_diagonal([rng], [T], dims...;
        block_weight=1, block_size=1,
        radius=nothing, return_sparse=false)

Creates a block‐diagonal matrix consisting of square blocks of size
`block_size` along the main diagonal [Ma2023](@cite).
Each block may be filled with
  - a single scalar
  - a vector of per‐block weights (length = number of blocks)

# Equations

```math
W_{i,j} =
\begin{cases}
    w_b, & \text{if }\left\lfloor\frac{i-1}{s}\right\rfloor =
        \left\lfloor\frac{j-1}{s}\right\rfloor = b,\;
           s = \text{block_size},\; b=0,\dots,nb-1, \\
    0,   & \text{otherwise,}
\end{cases}
```

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()` from
    WeightInitializers.
  - `T`: Element type of the matrix. Default is `Float32`.
  - `dims`: Dimensions of the output matrix (must be two-dimensional).

# Keyword arguments

  - `block_weight`:
    - scalar: every block is filled with that value
    - vector: length = number of blocks, one constant per block
    Default is `1.0`.
  - `block_size`: Size\(s\) of each square block on the diagonal. Default is `1.0`.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `return_sparse`: If `true`, returns the matrix as sparse.
    SparseArrays.jl must be lodead.
    Default is `false`.

# Examples

Changing the block size

```jldoctest blockdiag
julia> res_matrix = block_diagonal(10, 10; block_size=2)
10×10 Matrix{Float32}:
 1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0
```

Changing the weights, per block. Please note that you have to
know the number of blocks that you are going to have
(which usually is `res_size`/`block_size`).

```jldoctest blockdiag
julia> res_matrix = block_diagonal(10, 10; block_size=2, block_weight=[0.5, 2.0, -0.99, 1.0, -99.0])
10×10 Matrix{Float32}:
 0.5  0.5  0.0  0.0   0.0    0.0   0.0  0.0    0.0    0.0
 0.5  0.5  0.0  0.0   0.0    0.0   0.0  0.0    0.0    0.0
 0.0  0.0  2.0  2.0   0.0    0.0   0.0  0.0    0.0    0.0
 0.0  0.0  2.0  2.0   0.0    0.0   0.0  0.0    0.0    0.0
 0.0  0.0  0.0  0.0  -0.99  -0.99  0.0  0.0    0.0    0.0
 0.0  0.0  0.0  0.0  -0.99  -0.99  0.0  0.0    0.0    0.0
 0.0  0.0  0.0  0.0   0.0    0.0   1.0  1.0    0.0    0.0
 0.0  0.0  0.0  0.0   0.0    0.0   1.0  1.0    0.0    0.0
 0.0  0.0  0.0  0.0   0.0    0.0   0.0  0.0  -99.0  -99.0
 0.0  0.0  0.0  0.0   0.0    0.0   0.0  0.0  -99.0  -99.0
```
"""
function block_diagonal(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        block_weight::Union{Number, AbstractVector} = T(1),
        block_size::Integer = 1, radius::Union{AbstractFloat, Nothing} = nothing,
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    n_rows, n_cols = dims
    total = min(n_rows, n_cols)
    num_blocks = fld(total, block_size)
    remainder = total - num_blocks * block_size
    if remainder != 0
        @warn "\n
        With block_size=$block_size on a $n_rows×$n_cols matrix,
        only $num_blocks block(s) of size $block_size fit,
        leaving $remainder row(s)/column(s) unused.
        \n"
    end
    weights = isa(block_weight, AbstractVector) ? T.(block_weight) :
        fill(T(block_weight), num_blocks)
    @assert length(weights) == num_blocks "
      weight vector must have length = number of blocks
  "
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, n_rows, n_cols)
    for block in 1:num_blocks
        row_start = (block - 1) * block_size + 1
        row_end = row_start + block_size - 1
        col_start = (block - 1) * block_size + 1
        col_end = col_start + block_size - 1
        reservoir_matrix[row_start:row_end, col_start:col_end] .= weights[block]
    end
    scale_radius!(reservoir_matrix, radius)

    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    permutation_init([rng], [T], dims...;
        weight=0.1, permutation_matrix=nothing, return_sparse=false,
        radius=nothing, kwargs...)

Creates a permutation reservoir as described in [Boedecker2009](@cite), by first
initializing a scaled identity (self-loops) and then
applying a column permutation.

This construction yields:

```math
    \widehat{W} = \lambda P
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

  - `weight`: Weight used for the initial self-loop initialization (and the magnitude
    of the nonzeros after permutation). Default is 0.1.
  - `permutation_matrix`: Optional permutation matrix to apply. If `nothing`, a random
    permutation is generated (using `rng`) and applied.
  - `return_sparse`: flag for returning a `sparse` matrix.
    `true` requires `SparseArrays` to be loaded.
    Default is `false`.
  - `radius`: The desired spectral radius of the reservoir.
    If `nothing` is passed, no scaling takes place.
    Defaults to `nothing`.
  - `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
    If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
    `forward_weight` can be positive with a probability set by `positive_prob`. If set to
    `:irrational_sample!` the `weight` is negative if the decimal number of the
    irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
    assigned a negative sign after the chosen `strides`. `strides` can be a single
    number or an array. Default is `:no_sample`.
  - `positive_prob`: probability of the `weight` being positive when `sampling_type` is
    set to `:bernoulli_sample!`. Default is 0.5.
  - `irrational`: Irrational number whose decimals decide the sign of `weight`.
    Default is `pi`.
  - `start`: Which place after the decimal point the counting starts for the `irrational`
    sign counting. Default is 1.
  - `strides`: number of strides for assigning negative value to a weight. It can be an
    integer or an array. Default is 2.

## Examples

Default kwargs:

```jldoctest forcon
julia> reservoir_matrix = permutation_init(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.1
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0
```

Changing the weights magnitudes to a different unique value:

```jldoctest forcon
julia> reservoir_matrix = permutation_init(5, 5; weight=0.99)
5×5 Matrix{Float32}:
 0.0   0.0   0.0   0.0   0.99
 0.0   0.99  0.0   0.0   0.0
 0.99  0.0   0.0   0.0   0.0
 0.0   0.0   0.99  0.0   0.0
 0.0   0.0   0.0   0.99  0.0
```

Changing the weights signs with different sampling techniques:

```jldoctest forcon
julia> reservoir_matrix = permutation_init(5, 5; sampling_type=:bernoulli_sample!)
5×5 Matrix{Float32}:
 0.0  0.1   0.0  0.0   0.0
 0.0  0.0  -0.1  0.0   0.0
 0.1  0.0   0.0  0.0   0.0
 0.0  0.0   0.0  0.0  -0.1
 0.0  0.0   0.0  0.1   0.0
```

Changing the weights to random numbers. Note that the length of the given array
must be at least as long as the subdiagonal one wants to fill:

```jldoctest forcon
julia> reservoir_matrix = permutation_init(5, 5; weight=rand(Float32, 5))
5×5 Matrix{Float32}:
 0.0       0.0       0.0       0.0       0.0263106
 0.0       0.923927  0.0       0.0       0.0
 0.255075  0.0       0.0       0.0       0.0
 0.0       0.0       0.585589  0.0       0.0
 0.0       0.0       0.0       0.353418  0.0
```

Returning a sparse matrix:

```jldoctest forcon
julia> reservoir_matrix = permutation_init(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 5 stored entries:
  ⋅    ⋅    ⋅    ⋅   0.1
  ⋅   0.1   ⋅    ⋅    ⋅
 0.1   ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅   0.1   ⋅    ⋅
  ⋅    ⋅    ⋅   0.1   ⋅
```

"""
function permutation_init(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight = T(0.1), return_sparse::Bool = false,
        permutation_matrix::Union{Nothing, AbstractMatrix} = nothing,
        radius::Union{AbstractFloat, Nothing} = nothing,
        kwargs...
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    self_loop!(rng, reservoir_matrix, weight; kwargs...)
    permute_matrix!(rng, reservoir_matrix, permutation_matrix)
    scale_radius!(reservoir_matrix, radius)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    diagonal_init([rng], [T], dims...;
        return_sparse=false, weight=randn,
        kwargs...)

Creates a diagonal reservoir [Fette2005](@cite).

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

## Keyword arguments

- `weight`: Weight used for the initial self-loop initialization. Can be a single number,
  vector, or function to generate an array. Default is `randn`.
- `return_sparse`: flag for returning a `sparse` matrix.
  `true` requires `SparseArrays` to be loaded.
  Default is `false`.
- `return_diag`: flag for returning a `Diagonal` matrix. If both `return_diag`
  and `return_sparse` are set to `true` priority is given to `return_diag`.
  Default is `false`.
- `radius`: The desired spectral radius of the reservoir.
  If `nothing` is passed, no scaling takes place.
  Defaults to `nothing`.
- `sampling_type`: Sampling that decides the distribution of `weight` negative numbers.
  If set to `:no_sample` the sign is unchanged. If set to `:bernoulli_sample!` then each
  `forward_weight` can be positive with a probability set by `positive_prob`. If set to
  `:irrational_sample!` the `weight` is negative if the decimal number of the
  irrational number chosen is odd. If set to `:regular_sample!`, each weight will be
  assigned a negative sign after the chosen `strides`. `strides` can be a single
  number or an array. Default is `:no_sample`.
- `positive_prob`: probability of the `weight` being positive when `sampling_type` is
  set to `:bernoulli_sample!`. Default is 0.5.
- `irrational`: Irrational number whose decimals decide the sign of `weight`.
  Default is `pi`.
- `start`: Which place after the decimal point the counting starts for the `irrational`
  sign counting. Default is 1.
- `strides`: number of strides for assigning negative value to a weight. It can be an
  integer or an array. Default is 2.

## Examples

Default kwargs:

```jldoctest diaginit
julia> rr = diagonal_init(5, 5)
5×5 Matrix{Float32}:
 -0.359729  0.0       0.0      0.0      0.0
  0.0       1.08721   0.0      0.0      0.0
  0.0       0.0      -0.41959  0.0      0.0
  0.0       0.0       0.0      0.71891  0.0
  0.0       0.0       0.0      0.0      0.420247
```

Changing the weights magnitudes to a different unique value:

```jldoctest diaginit
julia> rr = diagonal_init(5, 5; weight=0.1)
5×5 Matrix{Float32}:
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0
 0.0  0.0  0.0  0.0  0.1
```

Changing the weights signs with different sampling techniques:

```jldoctest diaginit

```

Changing the weights to random numbers. Note that the length of the given array
must be at least as long as the subdiagonal one wants to fill:

```jldoctest diaginit
julia> rr = diagonal_init(5, 5; weight=0.1, sampling_type=:bernoulli_sample!)
5×5 Matrix{Float32}:
 0.1   0.0  0.0   0.0  0.0
 0.0  -0.1  0.0   0.0  0.0
 0.0   0.0  0.1   0.0  0.0
 0.0   0.0  0.0  -0.1  0.0
 0.0   0.0  0.0   0.0  0.1

julia> rr = diagonal_init(5, 5; weight=0.1, sampling_type=:irrational_sample!)
5×5 Matrix{Float32}:
 -0.1  0.0   0.0   0.0   0.0
  0.0  0.1   0.0   0.0   0.0
  0.0  0.0  -0.1   0.0   0.0
  0.0  0.0   0.0  -0.1   0.0
  0.0  0.0   0.0   0.0  -0.1
```

Returning a sparse matrix:

```jldoctest diaginit
julia> using SparseArrays

julia> rr = diagonal_init(5, 5; return_sparse=true)
5×5 SparseMatrixCSC{Float32, Int64} with 5 stored entries:
 -0.359729   ⋅         ⋅        ⋅        ⋅
   ⋅        1.08721    ⋅        ⋅        ⋅
   ⋅         ⋅       -0.41959   ⋅        ⋅
   ⋅         ⋅         ⋅       0.71891   ⋅
   ⋅         ⋅         ⋅        ⋅       0.420247
```

Returning a diagonal matrix:

```jldoctest diaginit
julia> rr = diagonal_init(5, 5; return_diag=true)
5×5 LinearAlgebra.Diagonal{Float32, Vector{Float32}}:
 -0.359729   ⋅         ⋅        ⋅        ⋅
   ⋅        1.08721    ⋅        ⋅        ⋅
   ⋅         ⋅       -0.41959   ⋅        ⋅
   ⋅         ⋅         ⋅       0.71891   ⋅
   ⋅         ⋅         ⋅        ⋅       0.420247
```

"""
function diagonal_init(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight = randn, return_sparse::Bool = false,
        return_diag::Bool = false, kwargs...
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    self_loop!(rng, reservoir_matrix, weight; kwargs...)
    if return_diag
        return Diagonal(diag(reservoir_matrix))
    else
        return return_init_as(Val(return_sparse), reservoir_matrix)
    end
end

### fallbacks
#fallbacks for initializers #eventually to remove once migrated to WeightInitializers.jl
for initializer in (
        :rand_sparse, :delay_line, :delayline_backward, :cycle_jumps,
        :simple_cycle, :pseudo_svd, :chaotic_init, :scaled_rand, :weighted_init,
        :weighted_minimal, :informed_init, :minimal_init, :chebyshev_mapping,
        :logistic_mapping, :modified_lm, :low_connectivity, :double_cycle, :selfloop_cycle,
        :selfloop_backward_cycle, :selfloop_delayline_backward, :selfloop_forwardconnection,
        :forward_connection, :true_doublecycle, :block_diagonal, :permutation_init,
        :diagonal_init,
    )
    @eval begin
        function ($initializer)(dims::Integer...; kwargs...)
            return $initializer(Utils.default_rng(), Float32, dims...; kwargs...)
        end
        function ($initializer)(rng::AbstractRNG, dims::Integer...; kwargs...)
            return $initializer(rng, Float32, dims...; kwargs...)
        end
        function ($initializer)(::Type{T}, dims::Integer...; kwargs...) where {T <: Number}
            return $initializer(Utils.default_rng(), T, dims...; kwargs...)
        end

        # Partial application
        function ($initializer)(rng::AbstractRNG; kwargs...)
            return PartialFunction.Partial{Nothing}($initializer, rng, kwargs)
        end
        function ($initializer)(::Type{T}; kwargs...) where {T <: Number}
            return PartialFunction.Partial{T}($initializer, nothing, kwargs)
        end
        function ($initializer)(rng::AbstractRNG, ::Type{T}; kwargs...) where {T <: Number}
            return PartialFunction.Partial{T}($initializer, rng, kwargs)
        end
        function ($initializer)(; kwargs...)
            return PartialFunction.Partial{Nothing}($initializer, nothing, kwargs)
        end
    end
end
