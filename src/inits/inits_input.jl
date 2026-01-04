### input layers
"""
    scaled_rand([rng], [T], dims...;
        scaling=0.1)

Create and return a matrix with random values, uniformly distributed within
a range defined by `scaling`.

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

## Keyword arguments

  - `scaling`: A scaling factor to define the range of the uniform distribution.
    The factor can be passed in three different ways:

      + A single number. In this case, the matrix elements will be randomly
        chosen from the range `[-scaling, scaling]`. Default option, with
        a the scaling value set to `0.1`.
      + A tuple `(lower, upper)`. The values define the range of the distribution.
        the matrix elements will be randomly created and scaled the range
        `[lower, upper]`.
      + A vector of length = `in_size`. In this case, the columns will be
        scaled individually by the entries of the vector. The entries can
        be numbers or tuples, which will mirror the behavior described above.

## Examples

Standard behavior with scaling given by a scalar:

```jldoctest scaledrand
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

Scaling with a tuple, providing lower and upper bound of the uniform distribution
from which the weights will be sampled:

```jldoctest scaledrand
julia> res_input = scaled_rand(8, 3, scaling = (0.1, 0.15))
8×3 Matrix{Float32}:
 0.108266  0.117683  0.120276
 0.128993  0.126018  0.106551
 0.131589  0.120211  0.142874
 0.120565  0.131978  0.147323
 0.12039   0.139184  0.129756
 0.148607  0.141981  0.128716
 0.105025  0.102723  0.11388
 0.100745  0.148357  0.133882
```

Scaling with a vector of scalars, where each provides the upper bound and its
negative provides the lower bound. Each column is scaled in order: first element
provides bounds for the first column, and so on:

```jldoctest
julia> res_input = scaled_rand(8, 3, scaling = [0.1, 0.2, 0.3])
8×3 Matrix{Float32}:
 -0.0669356  -0.0585384   -0.0566828
  0.0159724   0.00814199  -0.221385
  0.026355   -0.0383126    0.214489
 -0.0177412   0.0558246    0.267872
 -0.0184405   0.113474     0.0570667
  0.0944272   0.135849     0.0445941
 -0.0799005  -0.178218    -0.133435
 -0.0970182   0.186857     0.10659
```

Scaling with a vector of tuples, each providing both upper and lower bound.
Each column is scaled in order: first element
provides bounds for the first column, and so on:

```jldoctest scaledrand
julia> res_input = scaled_rand(8, 3, scaling = [(0.1, 0.2), (-0.2, -0.1), (0.3, 0.5)])
8×3 Matrix{Float32}:
 0.116532  -0.164635  0.381106
 0.157986  -0.147965  0.326205
 0.163177  -0.159578  0.471496
 0.141129  -0.136044  0.489291
 0.14078   -0.121632  0.419022
 0.197214  -0.116038  0.414865
 0.11005   -0.194554  0.355522
 0.101491  -0.103286  0.43553
```
"""
function scaled_rand(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling::Union{Number, Tuple, Vector} = T(0.1)
    ) where {T <: Number}
    res_size, in_size = dims
    layer_matrix = DeviceAgnostic.rand(rng, T, res_size, in_size)
    apply_scale!(layer_matrix, scaling, T)
    return layer_matrix
end

"""
    weighted_init([rng], [T], dims...;
        scaling=0.1, return_sparse=false)

Create and return a weighted input layer matrix. In this matrix, each
of the input signals `in_size` connects to the reservoir nodes
`res_size`/`in_size`. The nonzero entries are distributed uniformly
within a range defined by `scaling` [Lu2017](@cite).

Please note that this initializer computes its own reservoir size! If
the computed reservoir size is different than the provided one it will raise a
warning.

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

## Keyword arguments

  - `scaling`: A scaling factor to define the range of the uniform distribution.
    The factor can be passed in three different ways:

      + A single number. In this case, the matrix elements will be randomly
        chosen from the range `[-scaling, scaling]`. Default option, with
        a the scaling value set to `0.1`.

      + A tuple `(lower, upper)`. The values define the range of the distribution.
        the matrix elements will be randomly created and scaled the range
        `[lower, upper]`.
      + A vector of length = `in_size`. In this case, the columns will be
        scaled individually by the entries of the vector. The entries can
        be numbers or tuples, which will mirror the behavior described above.
      + `return_sparse`: flag for returning a `sparse` matrix.
        Default is `false`.

## Examples

Standard call with scaling provided by a scalar:

```jldoctest weightedinit
julia> res_input = weighted_init(9, 3; scaling = 0.1)
9×3 Matrix{Float32}:
  0.0452399   0.0         0.0
 -0.0348047   0.0         0.0
 -0.0386004   0.0         0.0
  0.0         0.0577838   0.0
  0.0        -0.0562827   0.0
  0.0         0.0441522   0.0
  0.0         0.0         0.00627948
  0.0         0.0        -0.0293777
  0.0         0.0        -0.0352914
```

Scaling with a tuple, providing lower and upper bound of the uniform distribution
from which the weights will be sampled:

```jldoctest weightedinit
julia> res_input = weighted_init(9, 3; scaling = (0.1, 0.5))
9×3 Matrix{Float32}:
 0.39048   0.0       0.0
 0.230391  0.0       0.0
 0.222799  0.0       0.0
 0.0       0.415568  0.0
 0.0       0.187435  0.0
 0.0       0.388304  0.0
 0.0       0.0       0.312559
 0.0       0.0       0.241245
 0.0       0.0       0.229417
```

Scaling with a vector of scalars, where each provides the upper bound and its
negative provides the lower bound. Each column is scaled in order: first element
provides bounds for the first column, and so on:

```jldoctest weightedinit
julia> res_input = weighted_init(9, 3; scaling = [0.1, 0.5, 0.9])
9×3 Matrix{Float32}:
  0.0452399   0.0        0.0
 -0.0348047   0.0        0.0
 -0.0386004   0.0        0.0
  0.0         0.288919   0.0
  0.0        -0.281413   0.0
  0.0         0.220761   0.0
  0.0         0.0        0.0565153
  0.0         0.0       -0.264399
  0.0         0.0       -0.317622
```

Scaling with a vector of tuples, each providing both upper and lower bound.
Each column is scaled in order: first element
provides bounds for the first column, and so on:

```jldoctest weightedinit
julia> res_input = weighted_init(9, 3; scaling = [(0.1, 0.2), (-0.2, -0.1), (0.3, 0.5)])
9×3 Matrix{Float32}:
 0.17262    0.0       0.0
 0.132598   0.0       0.0
 0.1307     0.0       0.0
 0.0       -0.121108  0.0
 0.0       -0.178141  0.0
 0.0       -0.127924  0.0
 0.0        0.0       0.40628
 0.0        0.0       0.370622
 0.0        0.0       0.364709
```

Example of matrix size change:

```jldoctest weightedinit
julia> res_input = weighted_init(8, 3)
┌ Warning: Reservoir size has changed!
│
│     Computed reservoir size (6) does not equal the provided reservoir size (8).
│
│     Using computed value (6). Make sure to modify the reservoir initializer accordingly.
│
└ @ ReservoirComputing ~/.julia/dev/ReservoirComputing/src/inits/inits_components.jl:20
6×3 Matrix{Float32}:
  0.0452399   0.0          0.0
 -0.0348047   0.0          0.0
  0.0        -0.0386004    0.0
  0.0         0.00981022   0.0
  0.0         0.0          0.0577838
  0.0         0.0         -0.0562827
```

Return sparse:

```jldoctest weightedinit
julia> using SparseArrays

julia> res_input = weighted_init(9, 3; return_sparse = true)
9×3 SparseMatrixCSC{Float32, Int64} with 9 stored entries:
  0.0452399    ⋅           ⋅
 -0.0348047    ⋅           ⋅
 -0.0386004    ⋅           ⋅
   ⋅          0.0577838    ⋅
   ⋅         -0.0562827    ⋅
   ⋅          0.0441522    ⋅
   ⋅           ⋅          0.00627948
   ⋅           ⋅         -0.0293777
   ⋅           ⋅         -0.0352914
```
"""
function weighted_init(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling::Union{Number, Tuple, Vector} = T(0.1),
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    approx_res_size, in_size = dims
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    check_modified_ressize(res_size, approx_res_size)

    layer_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    q = floor(Int, res_size / in_size)

    for idx in 1:in_size
        sc_rand = DeviceAgnostic.rand(rng, T, q)
        new_scaling = scaling isa AbstractVector ? scaling[idx] : scaling
        apply_scale!(sc_rand, new_scaling, T)
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

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

## Keyword arguments

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

## Examples

Standard call, changing the init weight:

```jldoctest weightedminimal
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
```

Random sign for each weight, drawn from a bernoulli distribution:

```jldoctest weightedminimal
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

Example of different reservoir size for the initializer:

```jldoctest weightedminimal
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
```
"""
function weighted_minimal(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Number = T(0.1), return_sparse::Bool = false,
        sampling_type = :no_sample, kwargs...
    ) where {T <: Number}
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

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

## Keyword arguments

  - `scaling`: The scaling factor for the input matrix.
    Default is 0.1.
  - `model_in_size`: The size of the input model.
  - `gamma`: The gamma value. Default is 0.5.

## Examples
"""
function informed_init(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling::Number = T(0.1), model_in_size::Integer,
        gamma::Number = T(0.5)
    ) where {T <: Number}
    res_size, in_size = dims
    state_size = in_size - model_in_size

    if state_size <= 0
        throw(DimensionMismatch("in_size must be greater than model_in_size"))
    end

    input_matrix = DeviceAgnostic.zeros(rng, T, res_size, in_size)
    zero_connections = DeviceAgnostic.zeros(rng, T, in_size)
    num_for_state = floor(Int, res_size * gamma)
    num_for_model = floor(Int, res_size * (1 - gamma))

    same_as_zero_row(row_index::Int) = zero_connections == @view(input_matrix[row_index, :])

    function zero_row_indices()
        zero_indices = Int[]
        for row_index in axes(input_matrix, 1)
            if same_as_zero_row(row_index)
                push!(zero_indices, row_index)
            end
        end
        return zero_indices
    end

    for _ in 1:num_for_state
        candidate_row_indices = zero_row_indices()
        isempty(candidate_row_indices) && break

        random_row_idx = rand(rng, candidate_row_indices)
        random_clm_idx = rand(rng, 1:state_size)

        input_matrix[random_row_idx, random_clm_idx] = (
            DeviceAgnostic.rand(rng, T) -
                T(0.5)
        ) * (T(2) * T(scaling))
    end

    for _ in 1:num_for_model
        candidate_row_indices = zero_row_indices()
        isempty(candidate_row_indices) && break

        random_row_idx = rand(rng, candidate_row_indices)
        random_clm_idx = rand(rng, (state_size + 1):in_size)

        input_matrix[random_row_idx, random_clm_idx] = (
            DeviceAgnostic.rand(rng, T) -
                T(0.5)
        ) * (T(2) * T(scaling))
    end

    return input_matrix
end

"""
    minimal_init([rng], [T], dims...;
        sampling_type=:bernoulli_sample!, weight=0.1, irrational=pi,
        start=1, p=0.5)

Create a dense matrix with same weights magnitudes determined by
`weight` [Rodan2011](@cite). The sign difference is randomly
determined by the `sampling` chosen.

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

## Keyword arguments

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

## Examples

Standard call:

```jldoctest minimalinit
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
```

Sampling weight sign from an irrational number:

```jldoctest minimalinit
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
```

Changing probability for the negative sign

```jldoctest minimalinit
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
function minimal_init(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        weight::Number = T(0.1), sampling_type::Symbol = :bernoulli_sample!,
        kwargs...
    ) where {T <: Number}
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

for j = 1, 2, …, n_cols (with n_cols typically equal to K+1, where K is the
number of input layer neurons). Subsequent rows are generated by
applying:

```math
    W[i+1, j] = \cos( \text{chebyshev_parameter} \cdot \acos(W[pi, j]))
```

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.
    `res_size` is assumed to be K+1.

## Keyword arguments

  - `amplitude`: Scaling factor used to initialize the first row.
    This parameter adjusts the amplitude of the sine function. Default value is one.
  - `sine_divisor`: Divisor applied in the sine function's phase. Default value is one.
  - `chebyshev_parameter`: Control parameter for the Chebyshev mapping in
    subsequent rows. This parameter influences the distribution of the
    matrix elements. Default is one.
  - `return_sparse`: If `true`, the function returns the matrix as a sparse matrix.
    Default is `false`.

## Examples

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
function chebyshev_mapping(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        amplitude::AbstractFloat = one(T), sine_divisor::AbstractFloat = one(T),
        chebyshev_parameter::AbstractFloat = one(T),
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    input_matrix = DeviceAgnostic.zeros(rng, T, dims...)
    n_rows, n_cols = dims[1], dims[2]

    for idx_cols in 1:n_cols
        input_matrix[1, idx_cols] = amplitude * sin(idx_cols * pi / (sine_divisor * n_cols))
    end
    for idx_rows in 2:n_rows
        for idx_cols in 1:n_cols
            input_matrix[idx_rows, idx_cols] = cos(
                chebyshev_parameter * acos(
                    input_matrix[
                        idx_rows - 1, idx_cols,
                    ]
                )
            )
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

for each input index `j`, with `in_size` being the number of columns
provided in `dims`. Subsequent rows are generated recursively using
the logistic map recurrence:

```math
    W[i+1, j] = \text{logistic_parameter} \cdot W(i, j) \cdot (1 - W[i, j])
```

## Arguments
  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

## Keyword arguments

  - `amplitude`: Scaling parameter used in the sine initialization of the
    first row. Default is 0.3.
  - `sine_divisor`: Parameter used to adjust the phase in the sine initialization.
    Default is 5.9.
  - `logistic_parameter`: The parameter in the logistic mapping recurrence that
    governs the dynamics. Default is 3.7.
  - `return_sparse`: If `true`, returns the resulting matrix as a sparse matrix.
    Default is `false`.

## Examples

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
function logistic_mapping(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        amplitude::AbstractFloat = 0.3, sine_divisor::AbstractFloat = 5.9,
        logistic_parameter::AbstractFloat = 3.7,
        return_sparse::Bool = false
    ) where {T <: Number}
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

## Arguments
  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `T`: Type of the elements in the reservoir matrix.
    Default is `Float32`.
  - `dims`: Dimensions of the matrix. Should follow `res_size x in_size`.

## Keyword arguments

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

## Examples

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
function modified_lm(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        factor::Integer, amplitude::AbstractFloat = 0.3,
        sine_divisor::AbstractFloat = 5.9, logistic_parameter::AbstractFloat = 2.35,
        return_sparse::Bool = false
    ) where {T <: Number}
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
        output_matrix[base_row, idx_col] = amplitude * sin(
            ((idx_col - 1) * pi) /
                (factor * num_columns * sine_divisor)
        )
        for jdx in 1:(factor - 1)
            current_row = base_row + jdx
            previous_value = output_matrix[current_row - 1, idx_col]
            output_matrix[current_row, idx_col] = logistic_parameter * previous_value *
                (1 - previous_value)
        end
    end

    return return_init_as(Val(return_sparse), output_matrix)
end
