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
