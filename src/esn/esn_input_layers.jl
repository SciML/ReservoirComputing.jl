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
matrix = scaled_rand(rng, Float64, (100, 50); scaling=0.2)
"""
function scaled_rand(
    rng::AbstractRNG,
    ::Type{T},
    dims::Integer...;
    scaling=T(0.1)
) where {T <: Number}

    res_size, in_size = dims
    layer_matrix = rand(rng, Uniform(-scaling, scaling), res_size, in_size)
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
input_layer = weighted_init(rng, Float64, (3, 300); scaling=0.2)
```
# References
[^Lu2017]: Lu, Zhixin, et al.
    "Reservoir observers: Model-free inference of unmeasured variables in chaotic systems."
    Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
"""
function weighted_init(rng::AbstractRNG, ::Type{T}, dims::Integer...; scaling=T(0.1)) where {T <: Number}

    in_size, approx_res_size = dims
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    layer_matrix = zeros(T, res_size, in_size)
    q = floor(Int, res_size / in_size)

    for i in 1:in_size
        layer_matrix[((i - 1) * q + 1):((i) * q), i] = rand(rng, Uniform(-scaling, scaling), q)
    end

    return layer_matrix
end
