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

    @assert length(dims) == 2, "The dimensions must define a matrix (e.g., (res_size, in_size))"

    res_size, in_size = dims
    layer_matrix = rand(rng, Uniform(-scaling, scaling), res_size, in_size)
    return layer_matrix
end
