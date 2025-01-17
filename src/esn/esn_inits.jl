### input layers
"""
    scaled_rand([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...;
        scaling=0.1)

Create and return a matrix with random values, uniformly distributed within
a range defined by `scaling`.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.
  - `scaling`: A scaling factor to define the range of the uniform distribution.
    The matrix elements will be randomly chosen from the
    range `[-scaling, scaling]`. Defaults to `0.1`.

# Examples

```jldoctest
julia> res_input = scaled_rand(8, 3)
8×3 Matrix{Float32}:
 -0.0669356  -0.0292692  -0.0188943
  0.0159724   0.004071   -0.0737949
  0.026355   -0.0191563   0.0714962
 -0.0177412   0.0279123   0.0892906
 -0.0184405   0.0567368   0.0190222
  0.0944272   0.0679244   0.0148647
 -0.0799005  -0.0891089  -0.0444782
 -0.0970182   0.0934286   0.03553
```
"""
function scaled_rand(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling=T(0.1)) where {T <: Number}
    res_size, in_size = dims
    layer_matrix = (DeviceAgnostic.rand(rng, T, res_size, in_size) .- T(0.5)) .*
                   (T(2) * scaling)
    return layer_matrix
end

"""
    weighted_init([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...;
        scaling=0.1)

Create and return a matrix representing a weighted input layer.
This initializer generates a weighted input matrix with random non-zero
elements distributed uniformly within the range [-`scaling`, `scaling`] [^Lu2017].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.
  - `scaling`: The scaling factor for the weight distribution.
    Defaults to `0.1`.

# Examples

```jldoctest
julia> res_input = weighted_init(8, 3)
6×3 Matrix{Float32}:
  0.0452399   0.0          0.0
 -0.0348047   0.0          0.0
  0.0        -0.0386004    0.0
  0.0         0.00981022   0.0
  0.0         0.0          0.0577838
  0.0         0.0         -0.0562827
```

[^Lu2017]: Lu, Zhixin, et al.
    "Reservoir observers: Model-free inference of unmeasured variables in
    chaotic systems."
    Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
"""
function weighted_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling=T(0.1)) where {T <: Number}
    approx_res_size, in_size = dims
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    layer_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    q = floor(Int, res_size / in_size)

    for i in 1:in_size
        layer_matrix[((i - 1) * q + 1):((i) * q), i] = (DeviceAgnostic.rand(rng, T, q) .-
                                                        T(0.5)) .* (T(2) * scaling)
    end

    return layer_matrix
end

"""
    informed_init([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...;
        scaling=0.1, model_in_size, gamma=0.5)

Create an input layer for informed echo state networks [^Pathak2018].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.
  - `scaling`: The scaling factor for the input matrix.
    Default is 0.1.
  - `model_in_size`: The size of the input model.
  - `gamma`: The gamma value. Default is 0.5.

# Examples

[^Pathak2018]: Pathak, Jaideep, et al. "Hybrid forecasting of chaotic processes:
    Using machine learning in conjunction with a knowledge-based model."
    Chaos: An Interdisciplinary Journal of Nonlinear Science 28.4 (2018).
"""
function informed_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling=T(0.1), model_in_size, gamma=T(0.5)) where {T <: Number}
    res_size, in_size = dims
    state_size = in_size - model_in_size

    if state_size <= 0
        throw(DimensionMismatch("in_size must be greater than model_in_size"))
    end

    input_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    zero_connections = DeviceAgnostic.zeros(rng, T, in_size)
    num_for_state = floor(Int, res_size * gamma)
    num_for_model = floor(Int, res_size * (1 - gamma))

    for i in 1:num_for_state
        idxs = findall(Bool[zero_connections .== input_matrix[i, :]
                            for i in 1:size(input_matrix, 1)])
        random_row_idx = idxs[DeviceAgnostic.rand(rng, T, 1:end)]
        random_clm_idx = range(1, state_size; step=1)[DeviceAgnostic.rand(rng, T, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = (DeviceAgnostic.rand(rng, T) -
                                                        T(0.5)) .* (T(2) * scaling)
    end

    for i in 1:num_for_model
        idxs = findall(Bool[zero_connections .== input_matrix[i, :]
                            for i in 1:size(input_matrix, 1)])
        random_row_idx = idxs[DeviceAgnostic.rand(rng, T, 1:end)]
        random_clm_idx = range(state_size + 1, in_size; step=1)[DeviceAgnostic.rand(
            rng, T, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = (DeviceAgnostic.rand(rng, T) -
                                                        T(0.5)) .* (T(2) * scaling)
    end

    return input_matrix
end

"""
    minimal_init([rng::AbstractRNG=Utils.default_rng()], [T=Float32], dims...;
        sampling_type=:bernoulli, weight=0.1, irrational=pi, start=1, p=0.5)

Create a layer matrix with uniform weights determined by `weight`. The sign difference
is randomly determined by the `sampling` chosen.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.
  - `weight`: The weight used to fill the layer matrix. Default is 0.1.
  - `sampling_type`: The sampling parameters used to generate the input matrix.
    Default is `:bernoulli`.
  - `irrational`: Irrational number chosen for sampling if `sampling_type=:irrational`.
    Default is `pi`.
  - `start`: Starting value for the irrational sample. Default is 1
  - `p`: Probability for the Bernoulli sampling. Lower probability increases negative
    value. Higher probability increases positive values. Default is 0.5

# Examples

```jldoctest
julia> res_input = minimal_init(8, 3)
8×3 Matrix{Float32}:
  0.1  -0.1   0.1
 -0.1   0.1   0.1
 -0.1  -0.1   0.1
 -0.1  -0.1  -0.1
  0.1   0.1   0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1   0.1
  0.1  -0.1   0.1

julia> res_input = minimal_init(8, 3; sampling_type=:irrational)
8×3 Matrix{Float32}:
 -0.1   0.1  -0.1
  0.1  -0.1  -0.1
  0.1   0.1  -0.1
  0.1   0.1   0.1
 -0.1  -0.1  -0.1
  0.1   0.1   0.1
  0.1   0.1  -0.1
 -0.1   0.1  -0.1

julia> res_input = minimal_init(8, 3; p=0.1) # lower p -> more negative signs
8×3 Matrix{Float32}:
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
  0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1

julia> res_input = minimal_init(8, 3; p=0.8)# higher p -> more positive signs
8×3 Matrix{Float32}:
  0.1   0.1  0.1
 -0.1   0.1  0.1
 -0.1   0.1  0.1
  0.1   0.1  0.1
  0.1   0.1  0.1
  0.1  -0.1  0.1
 -0.1   0.1  0.1
  0.1   0.1  0.1
```
"""
function minimal_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        sampling_type::Symbol=:bernoulli, weight::Number=T(0.1), irrational::Real=pi,
        start::Int=1, p::Number=T(0.5)) where {T <: Number}
    res_size, in_size = dims
    if sampling_type == :bernoulli
        layer_matrix = _create_bernoulli(p, res_size, in_size, weight, rng, T)
    elseif sampling_type == :irrational
        layer_matrix = _create_irrational(irrational,
            start,
            res_size,
            in_size,
            weight,
            rng,
            T)
    else
        error("Sampling type not allowed. Please use one of :bernoulli or :irrational")
    end
    return layer_matrix
end

function _create_bernoulli(p::Number, res_size::Int, in_size::Int, weight::Number,
        rng::AbstractRNG, ::Type{T}) where {T <: Number}
    input_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    for i in 1:res_size
        for j in 1:in_size
            if DeviceAgnostic.rand(rng, T) < p
                input_matrix[i, j] = weight
            else
                input_matrix[i, j] = -weight
            end
        end
    end
    return input_matrix
end

function _create_irrational(irrational::Irrational, start::Int, res_size::Int,
        in_size::Int, weight::Number, rng::AbstractRNG,
        ::Type{T}) where {T <: Number}
    setprecision(BigFloat, Int(ceil(log2(10) * (res_size * in_size + start + 1))))
    ir_string = string(BigFloat(irrational)) |> collect
    deleteat!(ir_string, findall(x -> x == '.', ir_string))
    ir_array = DeviceAgnostic.zeros(rng, T, length(ir_string))
    input_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)

    for i in 1:length(ir_string)
        ir_array[i] = parse(Int, ir_string[i])
    end

    for i in 1:res_size
        for j in 1:in_size
            random_number = DeviceAgnostic.rand(rng, T)
            input_matrix[i, j] = random_number < 0.5 ? -weight : weight
        end
    end

    return T.(input_matrix)
end

### reservoirs

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
        radius=T(1.0), sparsity=T(0.1), std=T(1.0),
        return_sparse::Bool=false) where {T <: Number}
    lcl_sparsity = T(1) - sparsity #consistency with current implementations
    reservoir_matrix = sparse_init(rng, T, dims...; sparsity=lcl_sparsity, std=std)
    rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= radius / rho_w
    if Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix)
        error("Sparsity too low for size of the matrix. Increase res_size or increase sparsity")
    end

    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
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
        weight=T(0.1), return_sparse::Bool=false) where {T <: Number}
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    @assert length(dims) == 2&&dims[1] == dims[2] "The dimensions
    must define a square matrix (e.g., (100, 100))"

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
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
        weight=T(0.1), fb_weight=T(0.2), return_sparse::Bool=false) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = weight
        reservoir_matrix[i, i + 1] = fb_weight
    end

    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
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
        jump_size::Int=3, return_sparse::Bool=false) where {T <: Number}
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

    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
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
        weight=T(0.1), return_sparse::Bool=false) where {T <: Number}
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    reservoir_matrix[1, dims[1]] = weight
    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
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
        reverse_sort::Bool=false, return_sparse::Bool=false) where {T <: Number}
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

    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
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

### fallbacks
#fallbacks for initializers #eventually to remove once migrated to WeightInitializers.jl
for initializer in (:rand_sparse, :delay_line, :delay_line_backward, :cycle_jumps,
    :simple_cycle, :pseudo_svd,
    :scaled_rand, :weighted_init, :informed_init, :minimal_init)
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
