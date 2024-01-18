function get_ressize(reservoir)
    return size(reservoir, 1)
end

"""
    rand_sparse(rng::AbstractRNG, ::Type{T}, dims::Integer...; radius=1.0, sparsity=0.1)

Create and return a random sparse reservoir matrix for use in Echo State Networks (ESNs). The matrix will be of size specified by `dims`, with specified `sparsity` and scaled spectral radius according to `radius`.

# Arguments
- `rng`: An instance of `AbstractRNG` for random number generation.
- `T`: The data type for the elements of the matrix.
- `dims`: Dimensions of the reservoir matrix.
- `radius`: The desired spectral radius of the reservoir. Defaults to 1.0.
- `sparsity`: The sparsity level of the reservoir matrix, controlling the fraction of zero elements. Defaults to 0.1.

# Returns
A matrix representing the random sparse reservoir.

# References
This type of reservoir initialization is commonly used in ESNs for capturing temporal dependencies in data.
"""
function rand_sparse(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        radius = T(1.0),
        sparsity = T(0.1)) where {T <: Number}
    reservoir_matrix = Matrix{T}(sprand(rng, dims..., sparsity))
    reservoir_matrix = T(2.0) .* (reservoir_matrix .- T(0.5))
    replace!(reservoir_matrix, T(-1.0) => T(0.0))
    rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= radius / rho_w
    if Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix)
        error("Sparsity too low for size of the matrix. Increase res_size or increase sparsity")
    end
    return reservoir_matrix
end

"""
    delay_line(rng::AbstractRNG, ::Type{T}, dims::Integer...; weight=0.1) where {T <: Number}

Create and return a delay line reservoir matrix for use in Echo State Networks (ESNs). A delay line reservoir is a deterministic structure where each unit is connected only to its immediate predecessor with a specified weight. This method is particularly useful for tasks that require specific temporal processing.

# Arguments
- `rng`: An instance of `AbstractRNG` for random number generation. This argument is not used in the current implementation but is included for consistency with other initialization functions.
- `T`: The data type for the elements of the matrix.
- `dims`: Dimensions of the reservoir matrix. Typically, this should be a tuple of two equal integers representing a square matrix.
- `weight`: The weight determines the absolute value of all connections in the reservoir. Defaults to 0.1.

# Returns
A delay line reservoir matrix with dimensions specified by `dims`. The matrix is initialized such that each element in the `i+1`th row and `i`th column is set to `weight`, and all other elements are zeros.

# Example
```julia
reservoir = delay_line(Float64, 100, 100; weight=0.2)
```

# References
This type of reservoir initialization is described in:
Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE Transactions on Neural Networks 22.1 (2010): 131-144.
"""
function delay_line(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        weight = T(0.1)) where {T <: Number}
    reservoir_matrix = zeros(T, dims...)
    @assert length(dims) == 2 && dims[1] == dims[2],
    "The dimensions must define a square matrix (e.g., (100, 100))"

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = weight
    end

    return reservoir_matrix
end

for initializer in (:rand_sparse, :delay_line)
    NType = ifelse(initializer === :rand_sparse, Real, Number)
    @eval function ($initializer)(dims::Integer...; kwargs...)
        return $initializer(_default_rng(), Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG, dims::Integer...; kwargs...)
        return $initializer(rng, Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(::Type{T},
            dims::Integer...; kwargs...) where {T <: $NType}
        return $initializer(_default_rng(), T, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG; kwargs...)
        return __partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval function ($initializer)(rng::AbstractRNG,
            ::Type{T}; kwargs...) where {T <: $NType}
        return __partial_apply($initializer, ((rng, T), (; kwargs...)))
    end
    @eval ($initializer)(; kwargs...) = __partial_apply($initializer, (; kwargs...))
end

# from WeightInitializers.jl, TODO @MartinuzziFrancesco consider importing package
function _default_rng()
    @static if VERSION >= v"1.7"
        return Xoshiro(1234)
    else
        return MersenneTwister(1234)
    end
end

__partial_apply(fn, inp) = fn$inp
