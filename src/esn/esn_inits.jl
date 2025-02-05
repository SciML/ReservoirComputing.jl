### input layers
"""
    scaled_rand([rng], [T], dims...;
        scaling=0.1)

Create and return a matrix with random values, uniformly distributed within
a range defined by `scaling`.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

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
    weighted_init([rng], [T], dims...;
        scaling=0.1, return_sparse=true)

Create and return a matrix representing a weighted input layer.
This initializer generates a weighted input matrix with random non-zero
elements distributed uniformly within the range [-`scaling`, `scaling`] [^Lu2017].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

  - `scaling`: The scaling factor for the weight distribution.
    Defaults to `0.1`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `true`.

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
        scaling=T(0.1), return_sparse::Bool=true) where {T <: Number}
    approx_res_size, in_size = dims
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    layer_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    q = floor(Int, res_size / in_size)

    for i in 1:in_size
        layer_matrix[((i - 1) * q + 1):((i) * q), i] = (DeviceAgnostic.rand(rng, T, q) .-
                                                        T(0.5)) .* (T(2) * scaling)
    end

    return return_sparse ? sparse(layer_matrix) : layer_matrix
end

"""
    informed_init([rng], [T], dims...;
        scaling=0.1, model_in_size, gamma=0.5)

Create an input layer for informed echo state networks [^Pathak2018].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

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
    minimal_init([rng], [T], dims...;
        sampling_type=:bernoulli, weight=0.1, irrational=pi, start=1, p=0.5)

Create a layer matrix with uniform weights determined by `weight`. The sign difference
is randomly determined by the `sampling` chosen.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

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

@doc raw"""
    chebyshev_mapping([rng], [T], dims...;
        amplitude=one(T), sine_divisor=one(T),
        chebyshev_parameter=one(T), return_sparse=true)

Generate a Chebyshev-mapped matrix [^xie2024]. The first row is initialized
using a sine function and subsequent rows are iteratively generated
via the Chebyshev mapping. The first row is defined as:

```math
    W[1, j] = \text{amplitude} \cdot \sin(j \cdot \pi / (\text{sine_divisor} 
        \cdot \text{n_cols}))
```

for j = 1, 2, …, n_cols (with n_cols typically equal to K+1, where K is the number of input layer neurons).
Subsequent rows are generated by applying the mapping:

```math
    W[i+1, j] = \cos( \text{chebyshev_parameter} \cdot \acos(W[pi, j]))
```

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.
    `res_size` is assumed to be K+1.

# Keyword arguments

  - `amplitude`: Scaling factor used to initialize the first row.
    This parameter adjusts the amplitude of the sine function. Default value is one.
  - `sine_divisor`: Divisor applied in the sine function's phase. Default value is one.
  - `chebyshev_parameter`: Control parameter for the Chebyshev mapping in
    subsequent rows. This parameter influences the distribution of the
    matrix elements. Default is one.
  - `return_sparse`: If `true`, the function returns the matrix as a sparse matrix. Default is `false`.

# Examples

```jldoctest
julia> input_matrix = chebyshev_mapping(10, 3)
10×3 Matrix{Float32}:
 0.866025  0.866025   1.22465f-16
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
 0.866025  0.866025  -4.37114f-8
```

[^xie2024]: Xie, Minzhi, Qianxue Wang, and Simin Yu.
    "Time Series Prediction of ESN Based on Chebyshev Mapping and Strongly
    Connected Topology."
    Neural Processing Letters 56.1 (2024): 30.
"""
function chebyshev_mapping(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        amplitude::AbstractFloat=one(T), sine_divisor::AbstractFloat=one(T),
        chebyshev_parameter::AbstractFloat=one(T),
        return_sparse::Bool=false) where {T <: Number}
    input_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    n_rows, n_cols = dims[1], dims[2]

    for idx_cols in 1:n_cols
        input_matrix[1, idx_cols] = amplitude * sin(idx_cols * pi / (sine_divisor * n_cols))
    end
    for idx_rows in 2:n_rows
        for idx_cols in 1:n_cols
            input_matrix[idx_rows, idx_cols] = cos(chebyshev_parameter * acos(input_matrix[
                idx_rows - 1, idx_cols]))
        end
    end

    return return_sparse ? sparse(input_matrix) : input_matrix
end

@doc raw"""
    logistic_mapping([rng], [T], dims...;
        amplitude=0.3, sine_divisor=5.9, logistic_parameter=3.7,
        return_sparse=true)

Generate an input weight matrix using a logistic mapping [^wang2022].The first
row is initialized using a sine function:

```math
    W[1, j] = \text{amplitude} \cdot \sin(j \cdot \pi / 
        (\text{sine_divisor} \cdot in_size))
```

for each input index `j`, with `in_size` being the number of columns provided in `dims`. Subsequent rows
are generated recursively using the logistic map recurrence:

```math
    W[i+1, j] = \text{logistic_parameter} \cdot W(i, j) \cdot (1 - W[i, j])
```

# Arguments
  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

  - `amplitude`: Scaling parameter used in the sine initialization of the
    first row. Default is 0.3.
  - `sine_divisor`: Parameter used to adjust the phase in the sine initialization.
    Default is 5.9.
  - `logistic_parameter`: The parameter in the logistic mapping recurrence that
    governs the dynamics. Default is 3.7.
  - `return_sparse`: If `true`, returns the resulting matrix as a sparse matrix.
    Default is `false`.

# Examples

```jldoctest
julia> logistic_mapping(8, 3)
8×3 Matrix{Float32}:
 0.0529682  0.104272  0.1523
 0.185602   0.345578  0.477687
 0.559268   0.836769  0.923158
 0.912003   0.50537   0.262468
 0.296938   0.924893  0.716241
 0.772434   0.257023  0.751987
 0.650385   0.70656   0.69006
 0.841322   0.767132  0.791346

```


[^wang2022]: Wang, Heshan, et al. "Echo state network with logistic
    mapping and bias dropout for time series prediction."
    Neurocomputing 489 (2022): 196-210.
"""
function logistic_mapping(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        amplitude::AbstractFloat=0.3, sine_divisor::AbstractFloat=5.9,
        logistic_parameter::AbstractFloat=3.7,
        return_sparse::Bool=false) where {T <: Number}
    input_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    num_rows, num_columns = dims[1], dims[2]
    for col in 1:num_columns
        input_matrix[1, col] = amplitude * sin(col * pi / (sine_divisor * num_columns))
    end
    for row in 2:num_rows
        for col in 1:num_columns
            previous_value = input_matrix[row - 1, col]
            input_matrix[row, col] = logistic_parameter * previous_value *
                                     (1 - previous_value)
        end
    end

    return return_sparse ? sparse(input_matrix) : input_matrix
end

@doc raw"""
    modified_lm([rng], [T], dims...;
        factor, amplitude=0.3, sine_divisor=5.9, logistic_parameter=2.35,
        return_sparse=true)

Generate a input weight matrix based on the logistic mapping [^viehweg2025]. The
matrix is built so that each input is transformed into a high-dimensional feature
space via a recursive logistic map. For each input, a chain of weights is generated
as follows:
- The first element of the chain is initialized using a sine function:

```math
      W[1,j] = \text{amplitude} \cdot \sin( (j \cdot \pi) / 
          (\text{factor} \cdot \text{n} \cdot \text{sine_divisor}) )
```
  where `j` is the index corresponding to the input and `n` is the number of inputs.

- Subsequent elements are recursively computed using the logistic mapping:

```math
      W[i+1,j] = \text{logistic_parameter} \cdot W[i,j] \cdot (1 - W[i,j])
```

The resulting matrix has dimensions `(factor * in_size) x in_size`, where
`in_size` corresponds to the number of columns provided in `dims`.
If the provided number of rows does not match `factor * in_size` 
the number of rows is overridden.

# Arguments
  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

  - `factor`: The number of logistic map iterations (chain length) per input,
    determining the number of rows per input.
  - `amplitude`: Scaling parameter A for the sine-based initialization of
    the first element in each logistic chain. Default is 0.3.
  - `sine_divisor`: Parameter B used to adjust the phase in the sine initialization.
    Default is 5.9.
  - `logistic_parameter`: The parameter r in the logistic recurrence that governs
    the chain dynamics. Default is 2.35.
  - `return_sparse`: If `true`, returns the resulting matrix as a sparse matrix.
    Default is `true`.

# Examples

```jldoctest
julia> modified_lm(20, 10; factor=2)
20×10 SparseArrays.SparseMatrixCSC{Float32, Int64} with 18 stored entries:
⎡⢠⠀⠀⠀⠀⎤
⎢⠀⢣⠀⠀⠀⎥
⎢⠀⠀⢣⠀⠀⎥
⎢⠀⠀⠀⢣⠀⎥
⎣⠀⠀⠀⠀⢣⎦

julia> modified_lm(12, 4; factor=3)
12×4 SparseArrays.SparseMatrixCSC{Float32, Int64} with 9 stored entries:
  ⋅    ⋅          ⋅          ⋅ 
  ⋅    ⋅          ⋅          ⋅ 
  ⋅    ⋅          ⋅          ⋅ 
  ⋅   0.0133075   ⋅          ⋅ 
  ⋅   0.0308564   ⋅          ⋅ 
  ⋅   0.070275    ⋅          ⋅ 
  ⋅    ⋅         0.0265887   ⋅ 
  ⋅    ⋅         0.0608222   ⋅ 
  ⋅    ⋅         0.134239    ⋅ 
  ⋅    ⋅          ⋅         0.0398177
  ⋅    ⋅          ⋅         0.0898457
  ⋅    ⋅          ⋅         0.192168

```

[^viehweg2025]: Viehweg, Johannes, Constanze Poll, and Patrick Mäder.
    "Deterministic Reservoir Computing for Chaotic Time Series Prediction."
    arXiv preprint arXiv:2501.15615 (2025).
"""
function modified_lm(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        factor::Integer, amplitude::AbstractFloat=0.3,
        sine_divisor::AbstractFloat=5.9, logistic_parameter::AbstractFloat=2.35,
        return_sparse::Bool=true) where {T <: Number}
    num_columns = dims[2]
    expected_num_rows = factor * num_columns
    if dims[1] != expected_num_rows
        @warn """\n
        Provided dims[1] ($(dims[1])) is not equal to factor*num_columns ($expected_num_rows).
        Overriding number of rows to $expected_num_rows. \n
        """
    end
    output_matrix = DeviceAgnostic.zeros(rng, T, expected_num_rows, num_columns)
    for col in 1:num_columns
        base_row = (col - 1) * factor + 1
        output_matrix[base_row, col] = amplitude * sin(((col - 1) * pi) /
                                           (factor * num_columns * sine_divisor))
        for j in 1:(factor - 1)
            current_row = base_row + j
            previous_value = output_matrix[current_row - 1, col]
            output_matrix[current_row, col] = logistic_parameter * previous_value *
                                              (1 - previous_value)
        end
    end

    return return_sparse ? sparse(output_matrix) : output_matrix
end

### reservoirs

"""
    rand_sparse([rng], [T], dims...;
        radius=1.0, sparsity=0.1, std=1.0, return_sparse=true)

Create and return a random sparse reservoir matrix.
The matrix will be of size specified by `dims`, with specified `sparsity`
and scaled spectral radius according to `radius`.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.
# Keyword arguments

  - `radius`: The desired spectral radius of the reservoir.
    Defaults to 1.0.
  - `sparsity`: The sparsity level of the reservoir matrix,
    controlling the fraction of zero elements. Defaults to 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `true`.

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
        return_sparse::Bool=true) where {T <: Number}
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
    delay_line([rng], [T], dims...;
        weight=0.1, return_sparse=true)

Create and return a delay line reservoir matrix [^Rodan2010].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `weight`: Determines the value of all connections in the reservoir.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `true`.

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
        weight=T(0.1), return_sparse::Bool=true) where {T <: Number}
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    @assert length(dims) == 2&&dims[1] == dims[2] "The dimensions
    must define a square matrix (e.g., (100, 100))"

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
end

"""
    delay_line_backward([rng], [T], dims...;
        weight=0.1, fb_weight=0.2, return_sparse=true)

Create a delay line backward reservoir with the specified by `dims` and weights.
Creates a matrix with backward connections as described in [^Rodan2010].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

  # Keyword arguments

  - `weight`: The weight determines the absolute value of
    forward connections in the reservoir. Default is 0.1
  - `fb_weight`: Determines the absolute value of backward connections
    in the reservoir. Default is 0.2
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `true`.

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
        weight=T(0.1), fb_weight=T(0.2), return_sparse::Bool=true) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = weight
        reservoir_matrix[i, i + 1] = fb_weight
    end

    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
end

"""
    cycle_jumps([rng], [T], dims...; 
        cycle_weight=0.1, jump_weight=0.1, jump_size=3, return_sparse=true)

Create a cycle jumps reservoir with the specified dimensions,
cycle weight, jump weight, and jump size.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `cycle_weight`:  The weight of cycle connections.
    Default is 0.1.
  - `jump_weight`: The weight of jump connections.
    Default is 0.1.
  - `jump_size`:  The number of steps between jump connections.
    Default is 3.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `true`.

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
        jump_size::Int=3, return_sparse::Bool=true) where {T <: Number}
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
    simple_cycle([rng], [T], dims...; 
        weight=0.1, return_sparse=true)

Create a simple cycle reservoir with the specified dimensions and weight.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

  # Keyword arguments

  - `weight`: Weight of the connections in the reservoir matrix.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `true`.

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
        weight=T(0.1), return_sparse::Bool=true) where {T <: Number}
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    reservoir_matrix[1, dims[1]] = weight
    return return_sparse ? sparse(reservoir_matrix) : reservoir_matrix
end

"""
    pseudo_svd([rng], [T], dims...; 
        max_value=1.0, sparsity=0.1, sorted=true, reverse_sort=false,
        return_sparse=true)

Returns an initializer to build a sparse reservoir matrix with the given
`sparsity` by using a pseudo-SVD approach as described in [^yang].

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

  # Keyword arguments

  - `max_value`: The maximum absolute value of elements in the matrix.
    Default is 1.0
  - `sparsity`: The desired sparsity level of the reservoir matrix.
    Default is 0.1
  - `sorted`: A boolean indicating whether to sort the singular values before
    creating the diagonal matrix. Default is `true`.
  - `reverse_sort`: A boolean indicating whether to reverse the sorted
    singular values. Default is `false`.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `true`.

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
        reverse_sort::Bool=false, return_sparse::Bool=true) where {T <: Number}
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

"""
    chaotic_init([rng], [T], dims...;
        extra_edge_probability=T(0.1), spectral_radius=one(T),
        return_sparse_matrix=true)

Construct a chaotic reservoir matrix using a digital chaotic system [^xie2024].

The matrix topology is derived from a strongly connected adjacency
matrix based on a digital chaotic system operating at finite precision.
If the requested matrix order does not exactly match a valid order the
closest valid order is used.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `extra_edge_probability`: Probability of adding extra random edges in
    the adjacency matrix to enhance connectivity. Default is 0.1.
  - `desired_spectral_radius`: The target spectral radius for the
    reservoir matrix. Default is one.
  - `return_sparse_matrix`: If `true`, the function returns the
    reservoir matrix as a sparse matrix. Default is `true`.

# Examples

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

[^xie2024]: Xie, Minzhi, Qianxue Wang, and Simin Yu.
    "Time Series Prediction of ESN Based on Chebyshev Mapping and Strongly
    Connected Topology."
    Neural Processing Letters 56.1 (2024): 30.
"""
function chaotic_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        extra_edge_probability::AbstractFloat=T(0.1), spectral_radius::AbstractFloat=one(T),
        return_sparse_matrix::Bool=true) where {T <: Number}
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
        rng, chosen_bit_precision; extra_edge_probability=extra_edge_probability)
    reservoir_matrix = random_weight_matrix .* adjacency_matrix
    current_spectral_radius = maximum(abs, eigvals(reservoir_matrix))
    if current_spectral_radius != 0
        reservoir_matrix .*= spectral_radius / current_spectral_radius
    end

    return return_sparse_matrix ? sparse(reservoir_matrix) : reservoir_matrix
end

function digital_chaotic_adjacency(rng::AbstractRNG, bit_precision::Integer;
        extra_edge_probability::AbstractFloat=0.1)
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

### fallbacks
#fallbacks for initializers #eventually to remove once migrated to WeightInitializers.jl
for initializer in (:rand_sparse, :delay_line, :delay_line_backward, :cycle_jumps,
    :simple_cycle, :pseudo_svd, :chaotic_init,
    :scaled_rand, :weighted_init, :informed_init, :minimal_init, :chebyshev_mapping,
    :logistic_mapping, :modified_lm)
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
