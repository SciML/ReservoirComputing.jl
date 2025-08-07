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
    The factor can be passed in three different ways:

      + A single number. In this case, the matrix elements will be randomly
        chosen from the range `[-scaling, scaling]`. Default option, with
        a the scaling value set to `0.1`.
      + A tuple `(lower, upper)`. The values define the range of the distribution.
      + A vector. In this case, the columns will be scaled individually by the
        entries of the vector. The entries can be numbers or tuples, which will mirror
        the behavior described above.

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

julia> tt = scaled_rand(5, 3, scaling = (0.1, 0.15))
5×3 Matrix{Float32}:
  0.13631   0.110929  0.116177
  0.116299  0.136038  0.119713
  0.11535   0.144712  0.110029
  0.127453  0.12657   0.147656
  0.139446  0.117656  0.104712
```

Example with vector:

```jldoctest
julia> tt = scaled_rand(5, 3, scaling = [0.1, 0.2, 0.3])
5×3 Matrix{Float32}:
  0.0452399   -0.112565   -0.105874
 -0.0348047    0.0883044  -0.0634468
 -0.0386004    0.157698   -0.179648
  0.00981022   0.012559    0.271875
  0.0577838   -0.0587553  -0.243451

julia> tt = scaled_rand(5, 3, scaling = [(0.1, 0.2), (-0.2, -0.1), (0.3, 0.5)])
5×3 Matrix{Float32}:
  0.17262   -0.178141  0.364709
  0.132598  -0.127924  0.378851
  0.1307    -0.110575  0.340117
  0.154905  -0.14686   0.490625
  0.178892  -0.164689  0.31885
```
"""
function scaled_rand(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling::Union{Number, Tuple, Vector} = T(0.1)) where {T <: Number}
    res_size, in_size = dims
    layer_matrix = DeviceAgnostic.rand(rng, T, res_size, in_size)
    apply_scale!(layer_matrix, scaling, T)
    return layer_matrix
end

function apply_scale!(input_matrix, scaling::Number, ::Type{T}) where {T}
    @. input_matrix = (input_matrix - T(0.5)) * (T(2) * T(scaling))
    return input_matrix
end

function apply_scale!(input_matrix,
        scaling::Tuple{<:Number, <:Number}, ::Type{T}) where {T}
    lower, upper = T(scaling[1]), T(scaling[2])
    @assert lower<upper "lower < upper required"
    scale = upper - lower
    @. input_matrix = input_matrix * scale + lower
    return input_matrix
end

function apply_scale!(input_matrix,
        scaling::AbstractVector, ::Type{T}) where {T <: Number}
    ncols = size(input_matrix, 2)
    @assert length(scaling)==ncols "need one scaling per column"
    for (idx, col) in enumerate(eachcol(input_matrix))
        apply_scale!(col, scaling[idx], T)
    end
    return input_matrix
end

"""
    weighted_init([rng], [T], dims...;
        scaling=0.1, return_sparse=false)

Create and return a matrix representing a weighted input layer.
This initializer generates a weighted input matrix with random non-zero
elements distributed uniformly within the range
[-`scaling`, `scaling`] [Lu2017](@cite).

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
    Default is `false`.

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
"""
function weighted_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling::Number = T(0.1), return_sparse::Bool = false) where {T <: Number}
    throw_sparse_error(return_sparse)
    approx_res_size, in_size = dims
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    check_modified_ressize(res_size, approx_res_size)
    layer_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    q = floor(Int, res_size / in_size)

    for idx in 1:in_size
        sc_rand = (DeviceAgnostic.rand(rng, T, q) .- T(0.5)) .* (T(2) * T(scaling))
        layer_matrix[((idx - 1) * q + 1):((idx) * q), idx] = sc_rand
    end

    return return_init_as(Val(return_sparse), layer_matrix)
end

"""
    weighted_minimal([rng], [T], dims...;
        weight=0.1, return_sparse=false,
        sampling_type=:no_sample)

Create and return a minimal weighted input layer matrix.
This initializer generates a weighted input matrix with equal, deterministic
elements in the same construction as [`weighted_minimal]`(@ref),
inspired by [Lu2017](@cite).

Please note that this initializer computes its own reservoir size! If
the computed reservoir size is different than the provided one it will raise a
warning.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

  - `weight`: The value for all the weights in the input matrix.
    Defaults to `0.1`.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> res_input = weighted_minimal(8, 3)
┌ Warning: Reservoir size has changed!
│
│     Computed reservoir size (6) does not equal the provided reservoir size (8).
│
│     Using computed value (6). Make sure to modify the reservoir initializer accordingly.
│
└ @ ReservoirComputing ~/.julia/dev/ReservoirComputing/src/esn/esn_inits.jl:159
6×3 Matrix{Float32}:
 0.1  0.0  0.0
 0.1  0.0  0.0
 0.0  0.1  0.0
 0.0  0.1  0.0
 0.0  0.0  0.1
 0.0  0.0  0.1

julia> res_input = weighted_minimal(9, 3; weight = 0.99)
9×3 Matrix{Float32}:
 0.99  0.0   0.0
 0.99  0.0   0.0
 0.99  0.0   0.0
 0.0   0.99  0.0
 0.0   0.99  0.0
 0.0   0.99  0.0
 0.0   0.0   0.99
 0.0   0.0   0.99
 0.0   0.0   0.99

julia> res_input = weighted_minimal(9, 3; sampling_type = :bernoulli_sample!)
9×3 Matrix{Float32}:
  0.1  -0.0  -0.0
 -0.1  -0.0  -0.0
  0.1  -0.0   0.0
 -0.0   0.1   0.0
  0.0   0.1  -0.0
  0.0   0.1   0.0
 -0.0  -0.0  -0.1
 -0.0  -0.0   0.1
  0.0  -0.0   0.1
```
"""
function weighted_minimal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Number = T(0.1), return_sparse::Bool = false,
        sampling_type = :no_sample, kwargs...) where {T <: Number}
    throw_sparse_error(return_sparse)
    approx_res_size, in_size = dims
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    check_modified_ressize(res_size, approx_res_size)
    layer_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    q = floor(Int, res_size / in_size)

    for idx in 1:in_size
        layer_matrix[((idx - 1) * q + 1):((idx) * q), idx] = T(weight) .* ones(T, q)
    end
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, layer_matrix; kwargs...)
    return return_init_as(Val(return_sparse), layer_matrix)
end

"""
    informed_init([rng], [T], dims...;
        scaling=0.1, model_in_size, gamma=0.5)

Create an input layer for informed echo state
networks [Pathak2018](@cite).

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
"""
function informed_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling::Number = T(0.1), model_in_size::Integer,
        gamma::Number = T(0.5)) where {T <: Number}
    res_size, in_size = dims
    state_size = in_size - model_in_size

    if state_size <= 0
        throw(DimensionMismatch("in_size must be greater than model_in_size"))
    end

    input_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    zero_connections = DeviceAgnostic.zeros(rng, T, in_size)
    num_for_state = floor(Int, res_size * gamma)
    num_for_model = floor(Int, res_size * (1 - gamma))

    for idx in 1:num_for_state
        idxs = findall(Bool[zero_connections .== input_matrix[jdx, :]
                            for jdx in axes(input_matrix, 1)])
        random_row_idx = idxs[DeviceAgnostic.rand(rng, T, 1:end)]
        random_clm_idx = range(1, state_size; step = 1)[DeviceAgnostic.rand(
            rng, T, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = (DeviceAgnostic.rand(rng, T) -
                                                        T(0.5)) .* (T(2) * T(scaling))
    end

    for idx in 1:num_for_model
        idxs = findall(Bool[zero_connections .== input_matrix[jdx, :]
                            for jdx in axes(input_matrix, 1)])
        random_row_idx = idxs[DeviceAgnostic.rand(rng, T, 1:end)]
        random_clm_idx = range(state_size + 1, in_size; step = 1)[DeviceAgnostic.rand(
            rng, T, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = (DeviceAgnostic.rand(rng, T) -
                                                        T(0.5)) .* (T(2) * T(scaling))
    end

    return input_matrix
end

"""
    minimal_init([rng], [T], dims...;
        sampling_type=:bernoulli_sample!, weight=0.1, irrational=pi,
        start=1, p=0.5)

Create a layer matrix with uniform weights determined by
`weight` [Rodan2011](@cite). The sign difference is randomly
determined by the `sampling` chosen.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

# Keyword arguments

  - `weight`: The weight used to fill the layer matrix. Default is 0.1.
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

julia> res_input = minimal_init(8, 3; sampling_type = :irrational)
8×3 Matrix{Float32}:
 -0.1   0.1  -0.1
  0.1  -0.1  -0.1
  0.1   0.1  -0.1
  0.1   0.1   0.1
 -0.1  -0.1  -0.1
  0.1   0.1   0.1
  0.1   0.1  -0.1
 -0.1   0.1  -0.1

julia> res_input = minimal_init(8, 3; p = 0.1) # lower p -> more negative signs
8×3 Matrix{Float32}:
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
  0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1
 -0.1  -0.1  -0.1

julia> res_input = minimal_init(8, 3; p = 0.8)# higher p -> more positive signs
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
        weight::Number = T(0.1), sampling_type::Symbol = :bernoulli_sample!,
        kwargs...) where {T <: Number}
    res_size, in_size = dims
    input_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    input_matrix .+= T(weight)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, input_matrix; kwargs...)
    return input_matrix
end

@doc raw"""
    chebyshev_mapping([rng], [T], dims...;
        amplitude=one(T), sine_divisor=one(T),
        chebyshev_parameter=one(T), return_sparse=false)

Generate a Chebyshev-mapped matrix [Xie2024](@cite).
The first row is initialized
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
  - `return_sparse`: If `true`, the function returns the matrix as a sparse matrix.
    Default is `false`.

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
"""
function chebyshev_mapping(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        amplitude::AbstractFloat = one(T), sine_divisor::AbstractFloat = one(T),
        chebyshev_parameter::AbstractFloat = one(T),
        return_sparse::Bool = false) where {T <: Number}
    throw_sparse_error(return_sparse)
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

    return return_init_as(Val(return_sparse), input_matrix)
end

@doc raw"""
    logistic_mapping([rng], [T], dims...;
        amplitude=0.3, sine_divisor=5.9, logistic_parameter=3.7,
        return_sparse=false)

Generate an input weight matrix using a logistic mapping [Wang2022](@cite)
The first row is initialized using a sine function:

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
"""
function logistic_mapping(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        amplitude::AbstractFloat = 0.3, sine_divisor::AbstractFloat = 5.9,
        logistic_parameter::AbstractFloat = 3.7,
        return_sparse::Bool = false) where {T <: Number}
    throw_sparse_error(return_sparse)
    input_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    num_rows, num_columns = dims[1], dims[2]
    for idx_col in 1:num_columns
        input_matrix[1, idx_col] = amplitude *
                                   sin(idx_col * pi / (sine_divisor * num_columns))
    end
    for idx_row in 2:num_rows
        for idx_col in 1:num_columns
            previous_value = input_matrix[idx_row - 1, idx_col]
            input_matrix[idx_row, idx_col] = logistic_parameter * previous_value *
                                             (1 - previous_value)
        end
    end

    return return_init_as(Val(return_sparse), input_matrix)
end

@doc raw"""
    modified_lm([rng], [T], dims...;
        factor, amplitude=0.3, sine_divisor=5.9, logistic_parameter=2.35,
        return_sparse=false)

Generate a input weight matrix based on the logistic mapping [Viehweg2025](@cite).
Thematrix is built so that each input is transformed into a high-dimensional feature
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
    Default is `false`.

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
"""
function modified_lm(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        factor::Integer, amplitude::AbstractFloat = 0.3,
        sine_divisor::AbstractFloat = 5.9, logistic_parameter::AbstractFloat = 2.35,
        return_sparse::Bool = false) where {T <: Number}
    throw_sparse_error(return_sparse)
    num_columns = dims[2]
    expected_num_rows = factor * num_columns
    if dims[1] != expected_num_rows
        @warn """\n
        Provided dims[1] ($(dims[1])) is not equal to factor*num_columns ($expected_num_rows).
        Overriding number of rows to $expected_num_rows. \n
        """
    end
    output_matrix = DeviceAgnostic.zeros(rng, T, expected_num_rows, num_columns)
    for idx_col in 1:num_columns
        base_row = (idx_col - 1) * factor + 1
        output_matrix[base_row, idx_col] = amplitude * sin(((idx_col - 1) * pi) /
                                               (factor * num_columns * sine_divisor))
        for jdx in 1:(factor - 1)
            current_row = base_row + jdx
            previous_value = output_matrix[current_row - 1, idx_col]
            output_matrix[current_row, idx_col] = logistic_parameter * previous_value *
                                                  (1 - previous_value)
        end
    end

    return return_init_as(Val(return_sparse), output_matrix)
end

### reservoirs

"""
    rand_sparse([rng], [T], dims...;
        radius=1.0, sparsity=0.1, std=1.0, return_sparse=false)

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
    Default is `false`.

# Examples

```jldoctest
julia> res_matrix = rand_sparse(5, 5; sparsity = 0.5)
5×5 Matrix{Float32}:
 0.0        0.0        0.0        0.0      0.0
 0.0        0.794565   0.0        0.26164  0.0
 0.0        0.0       -0.931294   0.0      0.553706
 0.723235  -0.524727   0.0        0.0      0.0
 1.23723    0.0        0.181824  -1.5478   0.465328
```
"""
function rand_sparse(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        radius::Number = T(1.0), sparsity::Number = T(0.1), std::Number = T(1.0),
        return_sparse::Bool = false) where {T <: Number}
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
    Default is `false`.
  - `return_diag`: flag for returning a `Diagonal` matrix. If both `return_diag`
    and `return_sparse` are set to `true` priority is given to `return_diag`.
    Default is `false`.

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
"""
function pseudo_svd(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        max_value::Number = T(1.0), sparsity::Number = 0.1, sorted::Bool = true,
        reverse_sort::Bool = false, return_sparse::Bool = false,
        return_diag::Bool = false) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = create_diag(rng, T, dims[1],
        T(max_value);
        sorted = sorted,
        reverse_sort = reverse_sort)
    tmp_sparsity = get_sparsity(reservoir_matrix, dims[1])

    while tmp_sparsity <= sparsity
        reservoir_matrix *= create_qmatrix(rng, T, dims[1],
            rand_range(rng, T, dims[1]),
            rand_range(rng, T, dims[1]),
            DeviceAgnostic.rand(rng, T) * T(2) - T(1))
        tmp_sparsity = get_sparsity(reservoir_matrix, dims[1])
    end

    if return_diag
        return Diagonal(reservoir_matrix)
    else
        return return_init_as(Val(return_sparse), reservoir_matrix)
    end
end

#hacky workaround for the moment
function rand_range(rng, T, n::Int)
    return Int(1 + floor(DeviceAgnostic.rand(rng, T) * n))
end

function create_diag(rng::AbstractRNG, ::Type{T}, dim::Number, max_value::Number;
        sorted::Bool = true, reverse_sort::Bool = false) where {T <: Number}
    diagonal_matrix = DeviceAgnostic.zeros(rng, T, dim, dim)
    if sorted == true
        if reverse_sort == true
            diagonal_values = sort(
                DeviceAgnostic.rand(rng, T, dim) .* max_value; rev = true)
            diagonal_values[1] = max_value
        else
            diagonal_values = sort(DeviceAgnostic.rand(rng, T, dim) .* max_value)
            diagonal_values[end] = max_value
        end
    else
        diagonal_values = DeviceAgnostic.rand(rng, T, dim) .* max_value
    end

    for idx in 1:dim
        diagonal_matrix[idx, idx] = diagonal_values[idx]
    end

    return diagonal_matrix
end

function create_qmatrix(rng::AbstractRNG, ::Type{T}, dim::Number,
        coord_i::Number, coord_j::Number, theta::Number) where {T <: Number}
    qmatrix = DeviceAgnostic.zeros(rng, T, dim, dim)

    for idx in 1:dim
        qmatrix[idx, idx] = 1.0
    end

    qmatrix[coord_i, coord_i] = cos(T(theta))
    qmatrix[coord_j, coord_j] = cos(T(theta))
    qmatrix[coord_i, coord_j] = -sin(T(theta))
    qmatrix[coord_j, coord_i] = sin(T(theta))
    return qmatrix
end

function get_sparsity(M, dim)
    return size(M[M .!= 0], 1) / (dim * dim - size(M[M .!= 0], 1)) #nonzero/zero elements
end

"""
    chaotic_init([rng], [T], dims...;
        extra_edge_probability=T(0.1), spectral_radius=one(T),
        return_sparse=false)

Construct a chaotic reservoir matrix using a digital chaotic system [Xie2024](@cite).

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
  - `return_sparse`: If `true`, the function returns the
    reservoir matrix as a sparse matrix. Default is `false`.

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
"""
function chaotic_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        extra_edge_probability::AbstractFloat = T(0.1), spectral_radius::AbstractFloat = one(T),
        return_sparse::Bool = false) where {T <: Number}
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
        rng, chosen_bit_precision; extra_edge_probability = extra_edge_probability)
    reservoir_matrix = random_weight_matrix .* adjacency_matrix
    current_spectral_radius = maximum(abs, eigvals(reservoir_matrix))
    if current_spectral_radius != 0
        reservoir_matrix .*= spectral_radius / current_spectral_radius
    end

    return return_init_as(Val(return_sparse), reservoir_matrix)
end

function digital_chaotic_adjacency(rng::AbstractRNG, bit_precision::Integer;
        extra_edge_probability::AbstractFloat = 0.1)
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
                     return_sparse = false, connected=false,
                     in_degree = 1, radius = 1.0, cut_cycle = false)

Construct an internal reservoir connectivity matrix with low connectivity.

This function creates a square reservoir matrix with the specified in-degree
for each node [Griffith2019](@cite). When `in_degree` is 1, the function can enforce
a fully connected cycle if `connected` is `true`;
otherwise, it generates a random connectivity pattern.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword Arguments

  - `return_sparse`: If `true`, the function returns the
    reservoir matrix as a sparse matrix. Default is `false`.
  - `connected`: For `in_degree == 1`, if `true` a connected cycle is enforced.
    Default is `false`.
  - `in_degree`: The number of incoming connections per node.
    Must not exceed the number of nodes. Default is 1.
  - `radius`: The desired spectral radius of the reservoir.
    Defaults to 1.0.
  - `cut_cycle`: If `true`, removes one edge from the cycle to cut it.
    Default is `false`.
"""
function low_connectivity(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        return_sparse::Bool = false, connected::Bool = false,
        in_degree::Integer = 1, kwargs...) where {T <: Number}
    check_res_size(dims...)
    res_size = dims[1]
    if in_degree > res_size
        error("""
            In-degree k (got k=$(in_degree)) cannot exceed number of nodes N=$(res_size)
        """)
    end
    if in_degree == 1
        reservoir_matrix = build_cycle(
            Val(connected), rng, T, res_size; in_degree = in_degree, kwargs...)
    else
        reservoir_matrix = build_cycle(
            Val(false), rng, T, res_size; in_degree = in_degree, kwargs...)
    end
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

function build_cycle(::Val{false}, rng::AbstractRNG, ::Type{T}, res_size::Int;
        in_degree::Integer = 1, radius::Number = T(1.0), cut_cycle::Bool = false) where {T <:
                                                                                         Number}
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

function build_cycle(::Val{true}, rng::AbstractRNG, ::Type{T}, res_size::Int;
        in_degree::Integer = 1, radius::Number = T(1.0), cut_cycle::Bool = false) where {T <:
                                                                                         Number}
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
        reservoir_matrix::AbstractMatrix{T}, rng::AbstractRNG) where {T <: Number}
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

"""
    delay_line([rng], [T], dims...;
        weight=0.1, return_sparse=false,
        kwargs...)

Create and return a delay line reservoir matrix [Rodan2011](@cite).

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `weight`: Determines the value of all connections in the reservoir.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1.
  - `shift`: delay line shift. Default is 1.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> res_matrix = delay_line(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = delay_line(5, 5; weight = 1)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
```
"""
function delay_line(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Union{Number, AbstractVector} = T(0.1), shift::Integer = 1,
        return_sparse::Bool = false, kwargs...) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    delay_line!(rng, reservoir_matrix, T.(weight), shift; kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

"""
    delay_line_backward([rng], [T], dims...;
        weight=0.1, fb_weight=0.1, return_sparse=false,
        delay_kwargs=(), fb_kwargs=())

Create a delay line backward reservoir with the specified by `dims` and weights.
Creates a matrix with backward connections as described in [Rodan2011](@cite).

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `weight`: The weight determines the absolute value of
    forward connections in the reservoir.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1

  - `fb_weight`: Determines the absolute value of backward connections
    in the reservoir.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1.
  - `fb_shift`: How far the backward connection will be from the diagonal.
    Default is 2.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> res_matrix = delay_line_backward(5, 5)
5×5 Matrix{Float32}:
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = delay_line_backward(Float16, 5, 5)
5×5 Matrix{Float16}:
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1
 0.0  0.0  0.0  0.1  0.0
```
"""
function delay_line_backward(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Union{Number, AbstractVector} = T(0.1),
        fb_weight::Union{Number, AbstractVector} = T(0.2), shift::Integer = 1,
        fb_shift::Integer = 1, return_sparse::Bool = false,
        delay_kwargs::NamedTuple = NamedTuple(),
        fb_kwargs::NamedTuple = NamedTuple()) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    delay_line!(rng, reservoir_matrix, T.(weight), shift; delay_kwargs...)
    backward_connection!(rng, reservoir_matrix, T.(fb_weight), fb_shift; fb_kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

"""
    cycle_jumps([rng], [T], dims...;
        cycle_weight=0.1, jump_weight=0.1, jump_size=3, return_sparse=false,
        cycle_kwargs=(), jump_kwargs=())

Create a cycle jumps reservoir [Rodan2012](@cite).

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

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
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> res_matrix = cycle_jumps(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.1  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = cycle_jumps(5, 5; jump_size = 2)
5×5 Matrix{Float32}:
 0.0  0.0  0.1  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.1  0.1  0.0  0.0  0.1
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.1  0.1  0.0
```
"""
function cycle_jumps(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1),
        jump_weight::Union{Number, AbstractVector} = T(0.1),
        jump_size::Integer = 3, return_sparse::Bool = false,
        cycle_kwargs::NamedTuple = NamedTuple(),
        jump_kwargs::NamedTuple = NamedTuple()) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    res_size = first(dims)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    simple_cycle!(rng, reservoir_matrix, T.(cycle_weight); cycle_kwargs...)
    add_jumps!(rng, reservoir_matrix, T.(jump_weight), jump_size; jump_kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

"""
    simple_cycle([rng], [T], dims...;
        weight=0.1, return_sparse=false,
        kwargs...)

Create a simple cycle reservoir [Rodan2011](@cite).

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `weight`: Weight of the connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> res_matrix = simple_cycle(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.0

julia> res_matrix = simple_cycle(5, 5; weight = 11)
5×5 Matrix{Float32}:
  0.0   0.0   0.0   0.0  11.0
 11.0   0.0   0.0   0.0   0.0
  0.0  11.0   0.0   0.0   0.0
  0.0   0.0  11.0   0.0   0.0
  0.0   0.0   0.0  11.0   0.0
```
"""
function simple_cycle(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Union{Number, AbstractVector} = T(0.1),
        return_sparse::Bool = false, kwargs...) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    simple_cycle!(rng, reservoir_matrix, T.(weight); kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

"""
    double_cycle([rng], [T], dims...;
        cycle_weight=0.1, second_cycle_weight=0.1,
        return_sparse=false)

Creates a double cycle reservoir [Fu2023](@cite).

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `cycle_weight`: Weight of the upper cycle connections in the reservoir matrix.
    Default is 0.1.
  - `second_cycle_weight`: Weight of the lower cycle connections in the reservoir matrix.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `false`.

# Examples

```jldoctest
julia> reservoir_matrix = double_cycle(5, 5; cycle_weight = 0.1, second_cycle_weight = 0.3)
5×5 Matrix{Float32}:
 0.0  0.3  0.0  0.0  0.3
 0.1  0.0  0.3  0.0  0.0
 0.0  0.1  0.0  0.3  0.0
 0.0  0.0  0.1  0.0  0.3
 0.1  0.0  0.0  0.1  0.0
```
"""
function double_cycle(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1),
        second_cycle_weight::Union{Number, AbstractVector} = T(0.1),
        return_sparse::Bool = false) where {T <: Number}
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
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

"""
    true_double_cycle([rng], [T], dims...;
        cycle_weight=0.1, second_cycle_weight=0.1,
        return_sparse=false)

Creates a true double cycle reservoir, ispired by [Fu2023](@cite),
with cycles built on the definition by [Rodan2011](@cite).

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `cycle_weight`: Weight of the upper cycle connections in the reservoir matrix.
    Default is 0.1.

  - `second_cycle_weight`: Weight of the lower cycle connections in the reservoir matrix.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> true_double_cycle(5, 5; cycle_weight = 0.1, second_cycle_weight = 0.3)
5×5 Matrix{Float32}:
 0.0  0.3  0.0  0.0  0.1
 0.1  0.0  0.3  0.0  0.0
 0.0  0.1  0.0  0.3  0.0
 0.0  0.0  0.1  0.0  0.3
 0.3  0.0  0.0  0.1  0.0
```
"""
function true_double_cycle(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1),
        second_cycle_weight::Union{Number, AbstractVector} = T(0.1),
        return_sparse::Bool = false, cycle_kwargs::NamedTuple = NamedTuple(),
        second_cycle_kwargs::NamedTuple = NamedTuple()) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    simple_cycle!(rng, reservoir_matrix, cycle_weight; cycle_kwargs...)
    reverse_simple_cycle!(
        rng, reservoir_matrix, second_cycle_weight; second_cycle_kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_cycle([rng], [T], dims...;
        cycle_weight=0.1, selfloop_weight=0.1,
        return_sparse=false, kwargs...)

Creates a simple cycle reservoir with the
addition of self loops [Elsarraj2019](@cite).

This architecture is referred to as TP1 in the original paper.

# Equations

```math
W_{i,j} =
\begin{cases}
    ll, & \text{if } i = j \\
    r, & \text{if } j = i - 1 \text{ for } i = 2 \dots N \\
    r, & \text{if } i = 1, j = N \\
    0, & \text{otherwise}
\end{cases}
```

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

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
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> reservoir_matrix = selfloop_cycle(5, 5)
5×5 Matrix{Float32}:
 0.1  0.0  0.0  0.0  0.1
 0.1  0.1  0.0  0.0  0.0
 0.0  0.1  0.1  0.0  0.0
 0.0  0.0  0.1  0.1  0.0
 0.0  0.0  0.0  0.1  0.1

julia> reservoir_matrix = selfloop_cycle(5, 5; weight=0.2, selfloop_weight=0.5)
5×5 Matrix{Float32}:
 0.5  0.0  0.0  0.0  0.2
 0.2  0.5  0.0  0.0  0.0
 0.0  0.2  0.5  0.0  0.0
 0.0  0.0  0.2  0.5  0.0
 0.0  0.0  0.0  0.2  0.5
```
"""
function selfloop_cycle(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0),
        return_sparse::Bool = false, kwargs...) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = simple_cycle(rng, T, dims...;
        weight = T.(cycle_weight), return_sparse = false)
    self_loop!(rng, reservoir_matrix, T.(selfloop_weight); kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_feedback_cycle([rng], [T], dims...;
        cycle_weight=0.1, selfloop_weight=0.1,
        return_sparse=false)

Creates a cycle reservoir with feedback connections on even neurons and
self loops on odd neurons [Elsarraj2019](@cite).

This architecture is referred to as TP2 in the original paper.

# Equations

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

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `cycle_weight`: Weight of the cycle connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `selfloop_weight`: Weight of the self loops in the reservoir matrix.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
    Default is `false`.

# Examples

```jldoctest
julia> reservoir_matrix = selfloop_feedback_cycle(5, 5)
5×5 Matrix{Float32}:
 0.1  0.1  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.1  0.1  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.1

julia> reservoir_matrix = selfloop_feedback_cycle(5, 5; self_loop_weight=0.5)
5×5 Matrix{Float32}:
 0.5  0.1  0.0  0.0  0.1
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.5  0.1  0.0
 0.0  0.0  0.1  0.0  0.0
 0.0  0.0  0.0  0.1  0.5
```
"""
function selfloop_feedback_cycle(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        cycle_weight::Union{Number, AbstractVector} = T(0.1f0),
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0),
        return_sparse::Bool = false) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = simple_cycle(rng, T, dims...;
        weight = T.(cycle_weight), return_sparse = false)
    for idx in axes(reservoir_matrix, 1)
        if isodd(idx)
            reservoir_matrix[idx, idx] = T.(selfloop_weight)
        end
    end
    for idx in (first(axes(reservoir_matrix, 1)) + 1):last(axes(reservoir_matrix, 1))
        if iseven(idx)
            reservoir_matrix[idx - 1, idx] = T.(cycle_weight)
        end
    end
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_delayline_backward([rng], [T], dims...;
        weight=0.1, selfloop_weight=0.1, fb_weight=0.1,
        fb_shift=2, return_sparse=false, fb_kwargs=(),
        selfloop_kwargs=(), delay_kwargs=())

Creates a reservoir based on a delay line with the addition of self loops and
backward connections shifted by one [Elsarraj2019](@cite).

This architecture is referred to as TP3 in the original paper.

# Equations

```math
W_{i,j} =
\begin{cases}
    ll, & \text{if } i = j \text{ for } i = 1 \dots N \\
    r, & \text{if } j = i - 1 \text{ for } i = 2 \dots N \\
    r, & \text{if } j = i - 2 \text{ for } i = 3 \dots N \\
    0, & \text{otherwise}
\end{cases}
```

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `weight`: Weight of the cycle connections in the reservoir matrix.
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
    Default is 2.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> reservoir_matrix = selfloop_delayline_backward(5, 5)
5×5 Matrix{Float32}:
 0.1  0.0  0.1  0.0  0.0
 0.1  0.1  0.0  0.1  0.0
 0.0  0.1  0.1  0.0  0.1
 0.0  0.0  0.1  0.1  0.0
 0.0  0.0  0.0  0.1  0.1

julia> reservoir_matrix = selfloop_delayline_backward(5, 5; weight=0.3)
5×5 Matrix{Float32}:
 0.1  0.0  0.3  0.0  0.0
 0.3  0.1  0.0  0.3  0.0
 0.0  0.3  0.1  0.0  0.3
 0.0  0.0  0.3  0.1  0.0
 0.0  0.0  0.0  0.3  0.1
```
"""
function selfloop_delayline_backward(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        shift::Integer = 1, fb_shift::Integer = 2,
        weight::Union{Number, AbstractVector} = T(0.1f0),
        fb_weight::Union{Number, AbstractVector} = weight,
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0),
        return_sparse::Bool = false, delay_kwargs::NamedTuple = NamedTuple(),
        fb_kwargs::NamedTuple = NamedTuple(),
        selfloop_kwargs::NamedTuple = NamedTuple()) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    self_loop!(rng, reservoir_matrix, T.(selfloop_weight); selfloop_kwargs...)
    delay_line!(rng, reservoir_matrix, T.(weight), shift; delay_kwargs...)
    backward_connection!(rng, reservoir_matrix, T.(fb_weight), fb_shift; fb_kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    selfloop_forward_connection([rng], [T], dims...;
        weight=0.1, selfloop_weight=0.1,
        return_sparse=false, selfloop_kwargs=(),
        delay_kwargs=())

Creates a reservoir based on a forward connection of weights between even nodes
with the addition of self loops [Elsarraj2019](@cite).

This architecture is referred to as TP4 in the original paper.

# Equations

```math
W_{i,j} =
\begin{cases}
    ll, & \text{if } i = j \text{ for } i = 1 \dots N \\
    r, & \text{if } j = i - 2 \text{ for } i = 3 \dots N \\
    0, & \text{otherwise}
\end{cases}
```

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `weight`: Weight of the cycle connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the cycle
    you want to populate.
    Default is 0.1.
  - `selfloop_weight`: Weight of the self loops in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the diagonal
    you want to populate.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> reservoir_matrix = selfloop_forward_connection(5, 5)
5×5 Matrix{Float32}:
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.1  0.0  0.1  0.0  0.0
 0.0  0.1  0.0  0.1  0.0
 0.0  0.0  0.1  0.0  0.1

julia> reservoir_matrix = selfloop_forward_connection(5, 5; weight=0.5)
5×5 Matrix{Float32}:
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.5  0.0  0.1  0.0  0.0
 0.0  0.5  0.0  0.1  0.0
 0.0  0.0  0.5  0.0  0.1
```
"""
function selfloop_forward_connection(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Union{Number, AbstractVector} = T(0.1f0),
        selfloop_weight::Union{Number, AbstractVector} = T(0.1f0), shift::Integer = 2,
        return_sparse::Bool = false, delay_kwargs::NamedTuple = NamedTuple(),
        selfloop_kwargs::NamedTuple = NamedTuple()) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    self_loop!(rng, reservoir_matrix, T.(selfloop_weight); selfloop_kwargs...)
    delay_line!(rng, reservoir_matrix, T.(weight), shift; delay_kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    forward_connection([rng], [T], dims...;
        weight=0.1, selfloop_weight=0.1,
        return_sparse=false)

Creates a reservoir based on a forward connection of weights [Elsarraj2019](@cite).

This architecture is referred to as TP5 in the original paper.

# Equations

```math
W_{i,j} =
\begin{cases}
    r, & \text{if } j = i - 2 \text{ for } i = 3 \dots N \\
    0, & \text{otherwise}
\end{cases}
```

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix.

# Keyword arguments

  - `weight`: Weight of the cycle connections in the reservoir matrix.
    This can be provided as a single value or an array. In case it is provided as an
    array please make sure that the length of the array matches the length of the sub-diagonal
    you want to populate.
    Default is 0.1.
  - `return_sparse`: flag for returning a `sparse` matrix.
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

# Examples

```jldoctest
julia> reservoir_matrix = forward_connection(5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.1  0.0  0.0  0.0  0.0
 0.0  0.1  0.0  0.0  0.0
 0.0  0.0  0.1  0.0  0.0

julia> reservoir_matrix = forward_connection(5, 5; weight=0.5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.5  0.0  0.0  0.0  0.0
 0.0  0.5  0.0  0.0  0.0
 0.0  0.0  0.5  0.0  0.0
```
"""
function forward_connection(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Union{Number, AbstractVector} = T(0.1f0), return_sparse::Bool = false,
        kwargs...) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    delay_line!(rng, reservoir_matrix, T.(weight), 2; kwargs...)
    return return_init_as(Val(return_sparse), reservoir_matrix)
end

@doc raw"""
    block_diagonal([rng], [T], dims...;
        weight=1, block_size=1,
        return_sparse=false)

Creates a block‐diagonal matrix consisting of square blocks of size
`block_size` along the main diagonal [Ma2023](@cite).
Each block may be filled with
  - a single scalar
  - a vector of per‐block weights (length = number of blocks)

# Equations

```math
W_{i,j} =
\begin{cases}
    w_b, & \text{if }\left\lfloor\frac{i-1}{s}\right\rfloor = \left\lfloor\frac{j-1}{s}\right\rfloor = b,\;
           s = \text{block\_size},\; b=0,\dots,nb-1, \\
    0,   & \text{otherwise,}
\end{cases}
```

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`.
  - `T`: Element type of the matrix. Default is `Float32`.
  - `dims`: Dimensions of the output matrix (must be two-dimensional).

# Keyword arguments

  - `weight`:
    - scalar: every block is filled with that value
    - vector: length = number of blocks, one constant per block
    Default is `1.0`.
  - `block_size`: Size\(s\) of each square block on the diagonal. Default is `1.0`.
  - `return_sparse`: If `true`, returns the matrix as sparse.
    SparseArrays.jl must be lodead.
    Default is `false`.

# Examples

```jldoctest
# 4×4 with two 2×2 blocks of 1.0
julia> W1 = block_diagonal(4, 4; block_size=2)
4×4 Matrix{Float32}:
 1.0  1.0  0.0  0.0
 1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0
 0.0  0.0  1.0  1.0

# per-block weights [0.5, 2.0]
julia> W2 = block_diagonal(4, 4; block_size=2, weight=[0.5, 2.0])
4×4 Matrix{Float32}:
 0.5  0.5  0.0  0.0
 0.5  0.5  0.0  0.0
 0.0  0.0  2.0  2.0
 0.0  0.0  2.0  2.0
```
"""
function block_diagonal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Union{Number, AbstractVector} = T(1),
        block_size::Integer = 1,
        return_sparse::Bool = false) where {T <: Number}
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
    weights = isa(weight, AbstractVector) ? T.(weight) : fill(T(weight), num_blocks)
    @assert length(weights)==num_blocks "
      weight vector must have length = number of blocks
  "
    reservoir_matrix = DeviceAgnostic.zeros(rng, T, n_rows, n_cols)
    for block in 1:num_blocks
        row_start = (block - 1) * block_size + 1
        row_end = row_start + block_size - 1
        col_start = (block - 1) * block_size + 1
        col_end = col_start + block_size - 1
        @inbounds reservoir_matrix[row_start:row_end, col_start:col_end] .= weights[block]
    end

    return return_init_as(Val(return_sparse), reservoir_matrix)
end

### fallbacks
#fallbacks for initializers #eventually to remove once migrated to WeightInitializers.jl
for initializer in (:rand_sparse, :delay_line, :delay_line_backward, :cycle_jumps,
    :simple_cycle, :pseudo_svd, :chaotic_init, :scaled_rand, :weighted_init,
    :weighted_minimal, :informed_init, :minimal_init, :chebyshev_mapping,
    :logistic_mapping, :modified_lm, :low_connectivity, :double_cycle, :selfloop_cycle,
    :selfloop_feedback_cycle, :selfloop_delayline_backward, :selfloop_forward_connection,
    :forward_connection, :true_double_cycle, :block_diagonal)
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
