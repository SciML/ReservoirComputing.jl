# dispatch over dense inits
function return_init_as(::Val{false}, layer_matrix::AbstractVecOrMat)
    return layer_matrix
end

# error for sparse inits with no SparseArrays.jl call
function throw_sparse_error(return_sparse::Bool)
    if return_sparse && !isdefined(Main, :SparseArrays)
        error("""\n
            Sparse output requested but SparseArrays.jl is not loaded.
            Please load it with:

                using SparseArrays\n
            """)
    end
end

function check_modified_ressize(res_size::Integer, approx_res_size::Integer)
    if res_size != approx_res_size
        @warn """Reservoir size has changed!\n
            Computed reservoir size ($res_size) does not equal the \
            provided reservoir size ($approx_res_size). \n 
            Using computed value ($res_size). Make sure to modify the \
            reservoir initializer accordingly. \n
        """
    end
end

function check_res_size(dims::Integer...)
    if length(dims) != 2 || dims[1] != dims[2]
        error("""\n
            Internal reservoir matrix must be square (e.g., (100, 100)).
            Got dims = $(dims)\n
        """)
    end
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
        error("""\n
            Sparsity too low for size of the matrix.
            Increase res_size or increase sparsity.\n
          """)
    end
    return reservoir_matrix
end

function no_sample(rng::AbstractRNG, vecormat::AbstractVecOrMat)
    return vecormat
end

function regular_sample!(rng::AbstractRNG, vecormat::AbstractVecOrMat;
        strides::Union{Integer, AbstractVector{<:Integer}} = 2)
    return _regular_sample!(rng, vecormat, strides)
end

function _regular_sample!(rng::AbstractRNG, vecormat::AbstractVecOrMat, strides::Integer)
    for idx in eachindex(vecormat)
        if idx % strides == 0
            vecormat[idx] = -vecormat[idx]
        end
    end
end

function _regular_sample!(
        rng::AbstractRNG, vecormat::AbstractVecOrMat, strides::AbstractVector{<:Integer})
    next_flip = strides[1]
    strides_idx = 1

    for idx in eachindex(vecormat)
        if idx == next_flip
            vecormat[idx] = -vecormat[idx]
            strides_idx = (strides_idx % length(strides)) + 1
            next_flip += strides[strides_idx]
        end
    end
end

function bernoulli_sample!(
        rng::AbstractRNG, vecormat::AbstractVecOrMat; positive_prob::Number = 0.5)
    for idx in eachindex(vecormat)
        if rand(rng) > positive_prob
            vecormat[idx] = -vecormat[idx]
        end
    end
end

#TODO: @MartinuzziFrancesco maybe change name here #wait, for sure change name here
function irrational_sample!(rng::AbstractRNG, vecormat::AbstractVecOrMat;
        irrational::Irrational = pi, start::Int = 1)
    total_elements = length(vecormat)
    setprecision(BigFloat, Int(ceil(log2(10) * (total_elements + start + 1))))
    ir_string = collect(string(BigFloat(irrational)))
    deleteat!(ir_string, findall(x -> x == '.', ir_string))
    ir_array = zeros(Int, length(ir_string))
    for idx in eachindex(ir_string)
        ir_array[idx] = parse(Int, ir_string[idx])
    end

    for idx in eachindex(vecormat)
        digit_index = start + idx
        if digit_index > length(ir_array)
            error("Not enough digits available. Increase precision or adjust start.")
        end
        if isodd(ir_array[digit_index])
            vecormat[idx] = -vecormat[idx]
        end
    end

    return vecormat
end

"""
    delay_line!([rng], reservoir_matrix, weight, shift;
        sampling_type=:no_sample, irrational=pi, start=1,
        p=0.5)

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
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> delay_line!(matrix, 5.0, 2)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 5.0  0.0  0.0  0.0  0.0
 0.0  5.0  0.0  0.0  0.0
 0.0  0.0  5.0  0.0  0.0

 julia> delay_line!(matrix, 5.0, 2; sampling_type=:bernoulli_sample!)
5×5 Matrix{Float32}:
 0.0   0.0  0.0  0.0  0.0
 0.0   0.0  0.0  0.0  0.0
 5.0   0.0  0.0  0.0  0.0
 0.0  -5.0  0.0  0.0  0.0
 0.0   0.0  5.0  0.0  0.0
```
"""
function delay_line!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number,
        shift::Integer; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1) - shift)
    return delay_line!(rng, reservoir_matrix, weights, shift; kwargs...)
end

function delay_line!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector,
        shift::Integer; sampling_type = :no_sample, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - shift)
        reservoir_matrix[idx + shift, idx] = weight[idx]
    end
    return reservoir_matrix
end

"""
    backward_connection!([rng], reservoir_matrix, weight, shift;
        sampling_type=:no_sample, irrational=pi, start=1,
        p=0.5)

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

julia> backward_connection!(matrix, 3.0, 1; sampling_type = :bernoulli_sample!)
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
        shift::Integer; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1) - shift)
    return backward_connection!(rng, reservoir_matrix, weights, shift; kwargs...)
end

function backward_connection!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector,
        shift::Integer; sampling_type = :no_sample, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - shift)
        reservoir_matrix[idx, idx + shift] = weight[idx]
    end
    return reservoir_matrix
end

"""
    simple_cycle!([rng], reservoir_matrix, weight;
        sampling_type=:no_sample, irrational=pi, start=1,
        p=0.5)

Adds a simple cycle in the `reservoir_matrix`, with given
`weight`. The `weight` can be a single number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a simple cycle. Can be either a single number
    or an array.

# Keyword arguments

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
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> simple_cycle!(matrix, 1.0; sampling_type = :irrational_sample!)
5×5 Matrix{Float32}:
  0.0  0.0   0.0   0.0  -1.0
 -1.0  0.0   0.0   0.0   0.0
  0.0  1.0   0.0   0.0   0.0
  0.0  0.0  -1.0   0.0   0.0
  0.0  0.0   0.0  -1.0   0.0
```
"""
function simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1))
    return simple_cycle!(rng, reservoir_matrix, weights; kwargs...)
end

function simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector;
        sampling_type = :no_sample, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - 1)
        reservoir_matrix[idx + 1, idx] = weight[idx]
    end
    reservoir_matrix[1, end] = weight[end]
    return reservoir_matrix
end

"""
    reverse_simple_cycle!([rng], reservoir_matrix, weight;
        sampling_type=:no_sample, irrational=pi, start=1,
        p=0.5)

Adds a reverse simple cycle in the `reservoir_matrix`, with given
`weight`. The `weight` can be a single number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a simple cycle. Can be either a single number
    or an array.

# Keyword arguments

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
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> reverse_simple_cycle!(matrix, 1.0; sampling_type = :regular_sample!)
5×5 Matrix{Float32}:
 0.0  -1.0  0.0   0.0  0.0
 0.0   0.0  1.0   0.0  0.0
 0.0   0.0  0.0  -1.0  0.0
 0.0   0.0  0.0   0.0  1.0
 1.0   0.0  0.0   0.0  0.0
```
"""
function reverse_simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1))
    return reverse_simple_cycle!(rng, reservoir_matrix, weights; kwargs...)
end

function reverse_simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector;
        sampling_type = :no_sample, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in (first(axes(reservoir_matrix, 1)) + 1):last(axes(reservoir_matrix, 1))
        reservoir_matrix[idx - 1, idx] = weight[idx]
    end
    reservoir_matrix[end, 1] = weight[end]
    return reservoir_matrix
end

"""
    add_jumps!([rng], reservoir_matrix, weight, jump_size;
        sampling_type=:no_sample, irrational=pi, start=1,
        positive_prob=0.5)

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
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> add_jumps!(matrix, 1.0)
5×5 Matrix{Float32}:
  0.0  0.0   1.0   0.0   0.0
  0.0  0.0   0.0   0.0   0.0
  1.0  0.0   0.0   0.0   0.0
  0.0  0.0   0.0   0.0   1.0
  0.0  0.0   1.0   0.0   0.0
```
"""
function add_jumps!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::Number, jump_size::Integer; kwargs...)
    weights = fill(
        weight, length(collect(1:jump_size:(size(reservoir_matrix, 1) - jump_size))))
    return add_jumps!(rng, reservoir_matrix, weights, jump_size; kwargs...)
end

function add_jumps!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::AbstractVector, jump_size::Integer;
        sampling_type = :no_sample, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    w_idx = 1
    for idx in 1:jump_size:(size(reservoir_matrix, 1) - jump_size)
        tmp = (idx + jump_size) % size(reservoir_matrix, 1)
        if tmp == 0
            tmp = size(reservoir_matrix, 1)
        end
        reservoir_matrix[idx, tmp] = weight[w_idx]
        reservoir_matrix[tmp, idx] = weight[w_idx]
        w_idx += 1
    end
    return reservoir_matrix
end

"""
    self_loop!([rng], reservoir_matrix, weight, jump_size;
        sampling_type=:no_sample, irrational=pi, start=1,
        positive_prob=0.5)

Adds jumps to a given `reservoir_matrix` with chosen `weight` and determined `jump_size`.
`weight` can be either a number or an array.

# Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()`
    from WeightInitializers.
  - `reservoir_matrix`: matrix to be changed.
  - `weight`: weight to add as a self loop. Can be either a single number
    or an array.

# Keyword arguments

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
julia> matrix = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> self_loop!(matrix, 1.0)
5×5 Matrix{Float32}:
  1.0  0.0   0.0   0.0   0.0
  0.0  1.0   0.0   0.0   0.0
  0.0  0.0   1.0   0.0   0.0
  0.0  0.0   0.0   1.0   0.0
  0.0  0.0   0.0   0.0   1.0
```
"""
function self_loop!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::Number; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1))
    return self_loop!(rng, reservoir_matrix, weights; kwargs...)
end

function self_loop!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::AbstractVector; sampling_type = :no_sample, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in axes(reservoir_matrix, 1)
        reservoir_matrix[idx, idx] = weight[idx]
    end
    return reservoir_matrix
end

for init_component in (:delay_line!, :add_jumps!, :backward_connection!,
    :simple_cycle!, :reverse_simple_cycle!, :self_loop!)
    @eval begin
        function ($init_component)(args...; kwargs...)
            return $init_component(Utils.default_rng(), args...; kwargs...)
        end
    end
end
