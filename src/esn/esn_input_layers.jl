"""
    scaled_rand(rng::AbstractRNG, ::Type{T}, dims::Integer...; scaling=T(0.1)) where {T <: Number}

Create and return a matrix with random values, uniformly distributed within a range defined by `scaling`. This function is useful for initializing matrices, such as the layers of a neural network, with scaled random values.

# Arguments

  - `rng`: An instance of `AbstractRNG` for random number generation.
  - `T`: The data type for the elements of the matrix.
  - `dims`: Dimensions of the matrix. It must be a 2-element tuple specifying the number of rows and columns (e.g., `(res_size, in_size)`).
  - `scaling`: A scaling factor to define the range of the uniform distribution. The matrix elements will be randomly chosen from the range `[-scaling, scaling]`. Defaults to `T(0.1)`.

# Returns

A matrix of type with dimensions specified by `dims`. Each element of the matrix is a random number uniformly distributed between `-scaling` and `scaling`.

# Example

```julia
rng = Random.default_rng()
matrix = scaled_rand(rng, Float64, (100, 50); scaling = 0.2)
```
"""
function scaled_rand(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        scaling = T(0.1)) where {T <: Number}
    res_size, in_size = dims
    layer_matrix = (DeviceAgnostic.rand(rng, T, res_size, in_size) .- T(0.5)) .*
                   (T(2) * scaling)
    return layer_matrix
end

"""
    weighted_init(rng::AbstractRNG, ::Type{T}, dims::Integer...; scaling=T(0.1)) where {T <: Number}

Create and return a matrix representing a weighted input layer for Echo State Networks (ESNs). This initializer generates a weighted input matrix with random non-zero elements distributed uniformly within the range [-`scaling`, `scaling`], inspired by the approach in [^Lu].

# Arguments

  - `rng`: An instance of `AbstractRNG` for random number generation.
  - `T`: The data type for the elements of the matrix.
  - `dims`: A 2-element tuple specifying the approximate reservoir size and input size (e.g., `(approx_res_size, in_size)`).
  - `scaling`: The scaling factor for the weight distribution. Defaults to `T(0.1)`.

# Returns

A matrix representing the weighted input layer as defined in [^Lu2017]. The matrix dimensions will be adjusted to ensure each input unit connects to an equal number of reservoir units.

# Example

```julia
rng = Random.default_rng()
input_layer = weighted_init(rng, Float64, (3, 300); scaling = 0.2)
```

# References

[^Lu2017]: Lu, Zhixin, et al.
    "Reservoir observers: Model-free inference of unmeasured variables in chaotic systems."
    Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
"""
function weighted_init(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        scaling = T(0.1)) where {T <: Number}
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
    informed_init(rng::AbstractRNG, ::Type{T}, dims::Integer...; scaling=T(0.1), model_in_size, gamma=T(0.5)) where {T <: Number}

Create a layer of a neural network.

# Arguments

  - `rng::AbstractRNG`: The random number generator.
  - `T::Type`: The data type.
  - `dims::Integer...`: The dimensions of the layer.
  - `scaling::T = T(0.1)`: The scaling factor for the input matrix.
  - `model_in_size`: The size of the input model.
  - `gamma::T = T(0.5)`: The gamma value.

# Returns

  - `input_matrix`: The created input matrix for the layer.

# Example

```julia
rng = Random.default_rng()
dims = (100, 200)
model_in_size = 50
input_matrix = informed_init(rng, Float64, dims; model_in_size = model_in_size)
```
"""
function informed_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling = T(0.1), model_in_size, gamma = T(0.5)) where {T <: Number}
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
        random_clm_idx = range(1, state_size, step = 1)[DeviceAgnostic.rand(rng, T, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = (DeviceAgnostic.rand(rng, T) -
                                                        T(0.5)) .* (T(2) * scaling)
    end

    for i in 1:num_for_model
        idxs = findall(Bool[zero_connections .== input_matrix[i, :]
                            for i in 1:size(input_matrix, 1)])
        random_row_idx = idxs[DeviceAgnostic.rand(rng, T, 1:end)]
        random_clm_idx = range(state_size + 1, in_size, step = 1)[DeviceAgnostic.rand(
            rng, T, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = (DeviceAgnostic.rand(rng, T) -
                                                        T(0.5)) .* (T(2) * scaling)
    end

    return input_matrix
end

"""
    irrational_sample_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
    weight = 0.1,
    sampling = IrrationalSample(; irrational = pi, start = 1)
    ) where {T <: Number}

Create a layer matrix using the provided random number generator and sampling parameters.

# Arguments

  - `rng::AbstractRNG`: The random number generator used to generate random numbers.
  - `dims::Integer...`: The dimensions of the layer matrix.
  - `weight`: The weight used to fill the layer matrix. Default is 0.1.
  - `sampling`: The sampling parameters used to generate the input matrix. Default is IrrationalSample(irrational = pi, start = 1).

# Returns

The layer matrix generated using the provided random number generator and sampling parameters.

# Example

```julia
using Random
rng = Random.default_rng()
dims = (3, 2)
weight = 0.5
layer_matrix = irrational_sample_init(rng, Float64, dims; weight = weight,
    sampling = IrrationalSample(irrational = sqrt(2), start = 1))
```
"""
function minimal_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        sampling_type::Symbol = :bernoulli,
        weight::Number = T(0.1),
        irrational::Real = pi,
        start::Int = 1,
        p::Number = T(0.5)) where {T <: Number}
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

function _create_bernoulli(p::Number,
        res_size::Int,
        in_size::Int,
        weight::Number,
        rng::AbstractRNG,
        ::Type{T}) where {T <: Number}
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

function _create_irrational(irrational::Irrational,
        start::Int,
        res_size::Int,
        in_size::Int,
        weight::Number,
        rng::AbstractRNG,
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
