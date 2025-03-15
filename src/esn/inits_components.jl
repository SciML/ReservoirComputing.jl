# dispatch over dense inits
function return_init_as(::Val{false}, layer_matrix::AbstractVecOrMat)
    return layer_matrix
end

# error for sparse inits with no SparseArrays.jl call
function throw_sparse_error(return_sparse::Bool)
    if return_sparse && !haskey(Base.loaded_modules, :SparseArrays)
        error("""\n
            Sparse output requested but SparseArrays.jl is not loaded.
            Please load it with:

                using SparseArrays\n
            """)
    end
end

## scale spectral radius
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

function default_sample!(rng::AbstractRNG, vecormat::AbstractVecOrMat)
    return
end

function bernoulli!(rng::AbstractRNG, vecormat::AbstractVecOrMat; p::Number=0.5)
    for idx in eachindex(vecormat)
        if rand(rng) > p
            vecormat[idx] = -vecormat[idx]
        end
    end
end

#TODO: @MartinuzziFrancesco maybe change name here #wait, for sure change name here
function irrational!(rng::AbstractRNG, vecormat::AbstractVecOrMat;
        irrational::Irrational=pi, start::Int=1)
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

function delay_line!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number,
        shift::Int; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1) - shift)
    delay_line!(rng, reservoir_matrix, weights, shift; kwargs...)
end

function delay_line!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector,
        shift::Int; sampling_type=:default_sample!, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - shift)
        reservoir_matrix[idx + shift, idx] = weight[idx]
    end
end

function backward_connection!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number,
        shift::Int; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1) - shift)
    backward_connection!(rng, reservoir_matrix, weights, shift; kwargs...)
end

function backward_connection!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector,
        shift::Int; sampling_type=:default_sample!, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - shift)
        reservoir_matrix[idx, idx + shift] = weight[idx]
    end
end

function simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::Number; kwargs...)
    weights = fill(weight, size(reservoir_matrix, 1))
    simple_cycle!(rng, reservoir_matrix, weights; kwargs...)
end

function simple_cycle!(
        rng::AbstractRNG, reservoir_matrix::AbstractMatrix, weight::AbstractVector;
        sampling_type=:default_sample!, kwargs...)
    f_sample = getfield(@__MODULE__, sampling_type)
    f_sample(rng, weight; kwargs...)
    for idx in first(axes(reservoir_matrix, 1)):(last(axes(reservoir_matrix, 1)) - 1)
        reservoir_matrix[idx + 1, idx] = weight[idx]
    end
    reservoir_matrix[1, end] = weight[end]
end

function add_jumps!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::Number, jump_size::Int; kwargs...)
    weights = fill(
        weight, length(collect(1:jump_size:(size(reservoir_matrix, 1) - jump_size))))
    add_jumps!(rng, reservoir_matrix, weights, jump_size; kwargs...)
end

function add_jumps!(rng::AbstractRNG, reservoir_matrix::AbstractMatrix,
        weight::AbstractVector, jump_size::Int;
        sampling_type=:default_sample!, kwargs...)
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
end
