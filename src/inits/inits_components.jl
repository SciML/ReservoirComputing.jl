function apply_scale!(input_matrix::AbstractArray, scaling::Number, ::Type{T}) where {T}
    @. input_matrix = (input_matrix - T(0.5)) * (T(2) * T(scaling))
    return input_matrix
end

function apply_scale!(
        input_matrix::AbstractArray,
        scaling::Tuple{<:Number, <:Number}, ::Type{T}
    ) where {T}
    lower, upper = T(scaling[1]), T(scaling[2])
    lower < upper || throw(ArgumentError("scaling tuple must satisfy lower < upper, got $scaling"))
    scale = upper - lower
    @. input_matrix = input_matrix * scale + lower
    return input_matrix
end

function apply_scale!(
        input_matrix::AbstractMatrix,
        scaling::AbstractVector, ::Type{T}
    ) where {T <: Number}
    ncols = size(input_matrix, 2)
    length(scaling) == ncols || throw(
        DimensionMismatch("need one scaling value per column, got $(length(scaling)) for $ncols columns")
    )
    for (idx, col) in enumerate(eachcol(input_matrix))
        apply_scale!(col, scaling[idx], T)
    end
    return input_matrix
end

@doc raw"""
    return_init_as(::Val{return_sparse}, initializer_output)

Convert an initializer output according to its `return_sparse` request.

!!! warning "Developer interface"
    This dispatch hook is for ReservoirComputing extension authors. End users
    should request sparse output through an initializer's `return_sparse`
    keyword rather than calling this function directly.

## Arguments

  - `return_sparse`: `Val(false)` for the built-in dense path or `Val(true)` for
    an extension-provided sparse representation.
  - `initializer_output`: an initializer result to return or convert.

## Extension contract

An extension that provides a sparse representation must define
`ReservoirComputing.return_init_as(::Val{true}, output)` for the output types it
supports. The method must return a representation with the same shape and values.
The built-in `Val(false)` method returns its input unchanged.

## Example

```julia
ReservoirComputing.return_init_as(Val(false), ones(2, 2)) == ones(2, 2)
```
"""
function return_init_as(::Val{false}, layer_matrix::AbstractVecOrMat)
    return layer_matrix
end

# error for sparse inits with no SparseArrays.jl call
function throw_sparse_error(return_sparse::Bool)
    return if return_sparse && !isdefined(Main, :SparseArrays)
        error(
            """\n
            Sparse output requested but SparseArrays.jl is not loaded.
            Please load it with:

                using SparseArrays\n
            """
        )
    end
end

function check_modified_ressize(res_size::Integer, approx_res_size::Integer)
    return if res_size != approx_res_size
        @warn """Reservoir size has changed!\n
            Computed reservoir size ($res_size) does not equal the \
        provided reservoir size ($approx_res_size). \n
            Using computed value ($res_size). Make sure to modify the \
        reservoir initializer accordingly. \n
        """
    end
end

function check_res_size(dims::Integer...)
    return if length(dims) != 2 || dims[1] != dims[2]
        throw(
            DimensionMismatch(
                "Internal reservoir matrix must be square (e.g., (100, 100)). Got dims = $(dims)"
            )
        )
    end
end

function check_inf_nan(weights::AbstractMatrix)
    has_nan = any(isnan, weights)
    has_inf = any(isinf, weights)
    if has_nan || has_inf
        throw(
            ArgumentError(
                "Created matrix contains invalid values (NaN=$has_nan, Inf=$has_inf)"
            )
        )
    end

    return nothing
end


## scale spectral radius
"""
    scale_radius!(matrix, radius)

Scale the spectral radius of the given matrix to be equal to the
given radius

# Arguments

  - `matrix`: Matrix to be scaled.
  - `radius`: desidered radius to scale the given matrix to
"""
function scale_radius!(reservoir_matrix::AbstractMatrix, radius::AbstractFloat)
    rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= radius / rho_w
    if Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix)
        error(
            """\n
              Sparsity too low for size of the matrix.
              Increase res_size or increase sparsity.\n
            """
        )
    end
    return reservoir_matrix
end

function scale_radius!(reservoir_matrix::AbstractMatrix, radius::Nothing)
    return reservoir_matrix
end

"""
    AbstractSignPattern

Abstract type for policies that modify the signs of initializer weights.

Implement `ReservoirComputing.__apply_signs!(rng, pattern, weights)` for custom
sign patterns. The method must mutate and return `weights`.
"""
abstract type AbstractSignPattern end

"""
    RandomSigns([positive_probability = 0.5])

Independently preserve each weight's sign with probability `positive_probability`
and flip it otherwise.
"""
struct RandomSigns <: AbstractSignPattern
    positive_probability::Float64

    function RandomSigns(positive_probability::Real = 0.5)
        0 <= positive_probability <= 1 || throw(
            ArgumentError("positive_probability must be between 0 and 1")
        )
        return new(Float64(positive_probability))
    end
end

"""
    RegularSigns([strides = 2])

Flip signs at positions determined by one stride or a repeating tuple of strides.
An integer stride `n` flips every `n`th weight. A tuple advances cumulatively by
each stride in a repeating cycle.
"""
struct RegularSigns{N} <: AbstractSignPattern
    strides::NTuple{N, Int}

    function RegularSigns(strides::NTuple{N, Int}) where {N}
        isempty(strides) && throw(ArgumentError("strides must not be empty"))
        all(>(0), strides) || throw(ArgumentError("all strides must be positive"))
        return new{N}(strides)
    end
end

RegularSigns() = RegularSigns(2)
RegularSigns(stride::Integer) = RegularSigns((Int(stride),))
RegularSigns(strides::Tuple{Vararg{Integer}}) = RegularSigns(Tuple(Int.(strides)))
RegularSigns(strides::AbstractVector{<:Integer}) = RegularSigns(Tuple(strides))

"""
    IrrationalDigitSigns([irrational = pi]; start = 1)

Flip weight signs where the corresponding decimal digit of `irrational` is odd,
starting at decimal position `start`.
"""
struct IrrationalDigitSigns{I <: Irrational} <: AbstractSignPattern
    irrational::I
    start::Int

    function IrrationalDigitSigns(
            irrational::I = pi; start::Integer = 1
        ) where {I <: Irrational}
        start >= 1 || throw(ArgumentError("start must be positive"))
        return new{I}(irrational, Int(start))
    end
end

__apply_signs!(rng::AbstractRNG, ::Nothing, weights::AbstractVecOrMat) = weights

function __apply_signs!(
        rng::AbstractRNG, pattern::RandomSigns, weights::AbstractVecOrMat
    )
    for idx in eachindex(weights)
        if rand(rng) > pattern.positive_probability
            weights[idx] = -weights[idx]
        end
    end
    return weights
end

function __apply_signs!(
        rng::AbstractRNG, pattern::RegularSigns, weights::AbstractVecOrMat
    )
    next_flip = first(pattern.strides)
    strides_idx = 1
    for idx in eachindex(weights)
        if idx == next_flip
            weights[idx] = -weights[idx]
            strides_idx = (strides_idx % length(pattern.strides)) + 1
            next_flip += pattern.strides[strides_idx]
        end
    end
    return weights
end

function __apply_signs!(
        rng::AbstractRNG, pattern::IrrationalDigitSigns, weights::AbstractVecOrMat
    )
    total_elements = length(weights)
    required_precision = Int(ceil(log2(10) * (total_elements + pattern.start + 1)))

    setprecision(BigFloat, required_precision) do
        irrational_string = string(BigFloat(pattern.irrational))
        irrational_digits = Int[]
        for character in irrational_string
            character == '.' && continue
            push!(irrational_digits, parse(Int, string(character)))
        end

        required_digits = pattern.start + total_elements
        length(irrational_digits) >= required_digits || throw(
            ArgumentError(
                "Not enough digits available. Increase precision or adjust start."
            )
        )

        for (element_index, storage_index) in enumerate(eachindex(weights))
            digit_index = pattern.start + element_index
            if isodd(irrational_digits[digit_index])
                weights[storage_index] = -weights[storage_index]
            end
        end
    end
    return weights
end

function __apply_signs_compat!(
        rng::AbstractRNG, signs::Union{Nothing, AbstractSignPattern},
        weights::AbstractVecOrMat,
        sampling_type::Nothing, kwargs
    )
    isempty(kwargs) || throw(
        ArgumentError(
            "sign-pattern keywords require a `signs` object; for example, use " *
                "`signs = RandomSigns(positive_probability)`"
        )
    )
    return __apply_signs!(rng, signs, weights)
end

function __apply_signs_compat!(
        rng::AbstractRNG, signs::Union{Nothing, AbstractSignPattern},
        weights::AbstractVecOrMat,
        sampling_type::Symbol, kwargs
    )
    isnothing(signs) || throw(
        ArgumentError("`signs` and deprecated `sampling_type` cannot be used together")
    )
    Base.depwarn(
        "`sampling_type` and its forwarded keywords are deprecated; pass a " *
            "`RandomSigns`, `RegularSigns`, or `IrrationalDigitSigns` object with " *
            "the `signs` keyword instead.",
        :sampling_type
    )
    sampler = getfield(@__MODULE__, sampling_type)
    return sampler(rng, weights; kwargs...)
end

function no_sample(rng::AbstractRNG, vecormat::AbstractVecOrMat)
    return __apply_signs!(rng, nothing, vecormat)
end

function regular_sample!(
        rng::AbstractRNG, vecormat::AbstractVecOrMat;
        strides::Union{Integer, AbstractVector{<:Integer}, Tuple{Vararg{Integer}}} = 2
    )
    return __apply_signs!(rng, RegularSigns(strides), vecormat)
end

function bernoulli_sample!(
        rng::AbstractRNG, vecormat::AbstractVecOrMat; positive_prob::Number = 0.5
    )
    return __apply_signs!(rng, RandomSigns(positive_prob), vecormat)
end

function irrational_sample!(
        rng::AbstractRNG, vecormat::AbstractVecOrMat;
        irrational::Irrational = pi, start::Int = 1
    )
    return __apply_signs!(rng, IrrationalDigitSigns(irrational; start), vecormat)
end

"""
    delay_line!([rng], reservoir_matrix, weight, shift; signs = nothing)

Adds a delay line in the `reservoir_matrix`, with given `shift` and
`weight`. The `weight` can be a single number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a delay line. Can be either a single number
    or an array.
  - `shift`: How far the delay line will be from the diagonal.

# Keyword arguments

  - `signs`: An `AbstractSignPattern` controlling sign flips. Use `RandomSigns`,
        `RegularSigns`, or `IrrationalDigitSigns`. Pass `nothing` to leave signs
        unchanged. Default is `nothing`.

# Examples

```jldoctest
julia> matrix = zeros(Float32, 5, 5);

julia> delay_line!(matrix, 5.0, 2);

julia> matrix[3, 1] == matrix[4, 2] == matrix[5, 3] == 5.0f0
true

julia> sampled_matrix = zeros(Float32, 5, 5);

julia> delay_line!(MersenneTwister(123), sampled_matrix, 5.0, 2;
           signs = RandomSigns());

julia> all(abs.(sampled_matrix[3:5, 1:3][diagind(sampled_matrix[3:5, 1:3])]) .== 5.0f0)
true
```
"""
function delay_line!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number,
        shift::Integer; kwargs...
    )
    weights = fill(weight, size(reservoir_matrix, 1) - shift)
    return delay_line!(rng, reservoir_matrix, weights, shift; kwargs...)
end

function delay_line!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector,
        shift::Integer;
        signs::Union{Nothing, AbstractSignPattern} = nothing,
        sampling_type = nothing,
        kwargs...
    )
    __apply_signs_compat!(rng, signs, weight, sampling_type, kwargs)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - shift)
        reservoir_matrix[idx + shift, idx] = weight[idx]
    end
    return reservoir_matrix
end

"""
    backward_connection!([rng], reservoir_matrix, weight, shift; signs = nothing)

Adds a backward connection in the `reservoir_matrix`, with given `shift` and
`weight`. The `weight` can be a single number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a backward connection. Can be either a single number
    or an array.
  - `shift`: How far the backward connection will be from the diagonal.

# Keyword arguments

  - `signs`: An `AbstractSignPattern` controlling sign flips. Use `RandomSigns`,
        `RegularSigns`, or `IrrationalDigitSigns`. Pass `nothing` to leave signs
        unchanged. Default is `nothing`.

# Examples

```jldoctest
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> backward_connection!(matrix, 3.0, 1)
5×5 Matrix{Float32}:
 0.0  3.0  0.0  0.0  0.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  3.0  0.0
 0.0  0.0  0.0  0.0  3.0
 0.0  0.0  0.0  0.0  0.0

julia> backward_connection!(matrix, 3.0, 1; signs = RandomSigns())
5×5 Matrix{Float32}:
 0.0  3.0   0.0  0.0   0.0
 0.0  0.0  -3.0  0.0   0.0
 0.0  0.0   0.0  3.0   0.0
 0.0  0.0   0.0  0.0  -3.0
 0.0  0.0   0.0  0.0   0.0
```
"""
function backward_connection!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number,
        shift::Integer; kwargs...
    )
    weights = fill(weight, size(reservoir_matrix, 1) - shift)
    return backward_connection!(rng, reservoir_matrix, weights, shift; kwargs...)
end

function backward_connection!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector,
        shift::Integer;
        signs::Union{Nothing, AbstractSignPattern} = nothing,
        sampling_type = nothing,
        kwargs...
    )
    __apply_signs_compat!(rng, signs, weight, sampling_type, kwargs)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - shift)
        reservoir_matrix[idx, idx + shift] = weight[idx]
    end
    return reservoir_matrix
end

"""
    simple_cycle!([rng], reservoir_matrix, weight; signs = nothing)

Adds a simple cycle in the `reservoir_matrix`, with given
`weight`. The `weight` can be a single number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a simple cycle. Can be either a single number
    or an array.

# Keyword arguments

  - `signs`: An `AbstractSignPattern` controlling sign flips. Use `RandomSigns`,
        `RegularSigns`, or `IrrationalDigitSigns`. Pass `nothing` to leave signs
        unchanged. Default is `nothing`.

# Examples

```jldoctest
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> simple_cycle!(matrix, 1.0; signs = IrrationalDigitSigns())
5×5 Matrix{Float32}:
  0.0  0.0   0.0   0.0  -1.0
 -1.0  0.0   0.0   0.0   0.0
  0.0  1.0   0.0   0.0   0.0
  0.0  0.0  -1.0   0.0   0.0
  0.0  0.0   0.0  -1.0   0.0
```
"""
function simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number; kwargs...
    )
    weights = fill(weight, size(reservoir_matrix, 1))
    return simple_cycle!(rng, reservoir_matrix, weights; kwargs...)
end

function simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector;
        signs::Union{Nothing, AbstractSignPattern} = nothing,
        sampling_type = nothing,
        kwargs...
    )
    __apply_signs_compat!(rng, signs, weight, sampling_type, kwargs)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - 1)
        reservoir_matrix[idx + 1, idx] = weight[idx]
    end
    reservoir_matrix[1, end] = weight[end]
    return reservoir_matrix
end

"""
    reverse_simple_cycle!([rng], reservoir_matrix, weight; signs = nothing)

Adds a reverse simple cycle in the `reservoir_matrix`, with given
`weight`. The `weight` can be a single number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a simple cycle. Can be either a single number
    or an array.

# Keyword arguments

  - `signs`: An `AbstractSignPattern` controlling sign flips. Use `RandomSigns`,
        `RegularSigns`, or `IrrationalDigitSigns`. Pass `nothing` to leave signs
        unchanged. Default is `nothing`.

# Examples

```jldoctest
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> reverse_simple_cycle!(matrix, 1.0; signs = RegularSigns())
5×5 Matrix{Float32}:
 0.0  -1.0  0.0   0.0  0.0
 0.0   0.0  1.0   0.0  0.0
 0.0   0.0  0.0  -1.0  0.0
 0.0   0.0  0.0   0.0  1.0
 1.0   0.0  0.0   0.0  0.0
```
"""
function reverse_simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number; kwargs...
    )
    weights = fill(weight, size(reservoir_matrix, 1))
    return reverse_simple_cycle!(rng, reservoir_matrix, weights; kwargs...)
end

function reverse_simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector;
        signs::Union{Nothing, AbstractSignPattern} = nothing,
        sampling_type = nothing,
        kwargs...
    )
    __apply_signs_compat!(rng, signs, weight, sampling_type, kwargs)
    for idx in (first(axes(reservoir_matrix, 1)) + 1):last(axes(reservoir_matrix, 1))
        reservoir_matrix[idx - 1, idx] = weight[idx]
    end
    reservoir_matrix[end, 1] = weight[end]
    return reservoir_matrix
end

"""
    add_jumps!([rng], reservoir_matrix, weight, jump_size;
        signs = nothing, start = 1)

Adds jumps to a given `reservoir_matrix` with chosen `weight` and determined `jump_size`.
`weight` can be either a number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a simple cycle. Can be either a single number
    or an array.
  - `jump_size`: size of the jump's distance.

# Keyword arguments

  - `signs`: An `AbstractSignPattern` controlling sign flips. Use `RandomSigns`,
        `RegularSigns`, or `IrrationalDigitSigns`. Pass `nothing` to leave signs
        unchanged. Default is `nothing`.

# Examples

```jldoctest
julia> matrix = zeros(Float32, 5, 5);

julia> add_jumps!(matrix, 1.0, 2);

julia> matrix[1, 3] == matrix[3, 1] == matrix[3, 5] == matrix[5, 3] == 1.0f0
true
```
"""
function add_jumps!(
        rng::AbstractRNG,
        reservoir_matrix::AbstractMatrix,
        weight::Number,
        jump_size::Integer;
        signs::Union{Nothing, AbstractSignPattern} = nothing,
        sampling_type = nothing,
        start::Integer = 1,
        kwargs...
    )
    N = size(reservoir_matrix, 1)
    g = gcd(N, jump_size)
    ring_len = (N % jump_size == 0) ? div(N, g) : fld(N, jump_size)
    weights = fill(weight, ring_len)
    return add_jumps!(
        rng, reservoir_matrix, weights, jump_size;
        signs, sampling_type, start, kwargs...
    )
end

function add_jumps!(
        rng::AbstractRNG,
        reservoir_matrix::AbstractMatrix,
        weight::AbstractVector,
        jump_size::Integer;
        signs::Union{Nothing, AbstractSignPattern} = nothing,
        sampling_type = nothing,
        start::Integer = 1,
        kwargs...
    )
    N = size(reservoir_matrix, 1)
    @assert N == size(reservoir_matrix, 2) "reservoir_matrix must be square"
    @assert 1 ≤ start ≤ N "start must be in 1:N"
    @assert 1 ≤ jump_size < N "jump_size must be in 1:(N-1)"

    divisible = (N % jump_size == 0)

    seq = Int[start]
    cur = start
    while true
        nxt = cur + jump_size
        if nxt > N
            if divisible
                nxt = ((cur + jump_size - 1) % N) + 1
                if nxt == start
                    break
                end
                push!(seq, nxt)
                cur = nxt
                continue
            else
                break
            end
        else
            push!(seq, nxt)
            cur = nxt
        end
    end

    num_edges = divisible ? length(seq) : (length(seq) - 1)
    @assert num_edges ≥ 0
    w = collect(weight)
    if length(w) < num_edges
        append!(w, fill(last(w), num_edges - length(w)))
    elseif length(w) > num_edges
        resize!(w, num_edges)
    end
    __apply_signs_compat!(rng, signs, w, sampling_type, kwargs)

    for k in 1:num_edges
        i = seq[k]
        j = divisible ? (k == num_edges ? start : seq[k + 1]) : seq[k + 1]
        wk = w[k]
        reservoir_matrix[i, j] = wk
        reservoir_matrix[j, i] = wk
    end

    return reservoir_matrix
end

"""
    self_loop!([rng], reservoir_matrix, weight; signs = nothing)

Adds jumps to a given `reservoir_matrix` with chosen `weight` and determined `jump_size`.
`weight` can be either a number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a self loop. Can be either a single number
    or an array.

# Keyword arguments

  - `signs`: An `AbstractSignPattern` controlling sign flips. Use `RandomSigns`,
        `RegularSigns`, or `IrrationalDigitSigns`. Pass `nothing` to leave signs
        unchanged. Default is `nothing`.

# Examples

```jldoctest
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> self_loop!(matrix, 1.0);

julia> diag(matrix) == fill(1.0f0, 5)
true
```
"""
function self_loop!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::Number; kwargs...
    )
    weights = fill(weight, size(reservoir_matrix, 1))
    return self_loop!(rng, reservoir_matrix, weights; kwargs...)
end

function self_loop!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::AbstractVector;
        signs::Union{Nothing, AbstractSignPattern} = nothing,
        sampling_type = nothing,
        kwargs...
    )
    __apply_signs_compat!(rng, signs, weight, sampling_type, kwargs)
    for idx in axes(reservoir_matrix, 1)
        reservoir_matrix[idx, idx] = weight[idx]
    end
    return reservoir_matrix
end

function self_loop!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight; kwargs...
    )
    weights = weight(rng, size(reservoir_matrix, 1))
    return self_loop!(rng, reservoir_matrix, weights; kwargs...)
end

@doc raw"""
    permute_matrix!([rng], reservoir_matrix,
        permutation_matrix=nothing)

Right-multiply `reservoir_matrix` by a permutation matrix to permute its columns.
The update overwrites the contents of `reservoir_matrix`.

If `permutation_matrix` is `nothing`, a random permutation is generated and converted
to a permutation matrix.

## Arguments

  - `rng`: Random number generator used when `permutation_matrix === nothing`.
    Default is typically `Utils.default_rng()` from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers)
    (if you provide a wrapper method without `rng`).
  - `reservoir_matrix`: The reservoir weight matrix to be permuted.
  - `permutation_matrix`: A square permutation matrix of matching size. If `nothing`,
    a random permutation is used.
"""
function permute_matrix!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix{T},
        permutation_matrix::Union{Nothing, AbstractMatrix} = nothing
    ) where {T}
    if permutation_matrix === nothing
        perm_array = randperm(rng, size(reservoir_matrix, 1))
        permutation_matrix = create_permutation_matrix(perm_array, reservoir_matrix)
    end
    t_pm = eltype(permutation_matrix) === T ? permutation_matrix : T.(permutation_matrix)
    tmp = similar(reservoir_matrix)
    mul!(tmp, reservoir_matrix, t_pm)
    copyto!(reservoir_matrix, tmp)
    return reservoir_matrix
end

function create_permutation_matrix(perm_array::AbstractVector{Int}, reservoir_matrix)
    num_perm = length(perm_array)
    T = eltype(reservoir_matrix)
    permutation_matrix = similar(reservoir_matrix, num_perm, num_perm)
    fill!(permutation_matrix, zero(T))
    for idx in eachindex(perm_array)
        permutation_matrix[perm_array[idx], idx] = one(T)
    end
    return permutation_matrix
end

for init_component in (
        :delay_line!, :add_jumps!, :backward_connection!,
        :simple_cycle!, :reverse_simple_cycle!, :self_loop!,
    )
    @eval begin
        function ($init_component)(args...; kwargs...)
            return $init_component(Utils.default_rng(), args...; kwargs...)
        end
    end
end
