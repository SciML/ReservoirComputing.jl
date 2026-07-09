@testsetup module GenericTestSetup

using Random
using Test
using ReservoirComputing: setup

export dense_init, vector_init, typed_inputs
export common_model_kwargs, res_model_kwargs, es2n_model_kwargs, run_model_smoke

dense_init(::Type{T}; value = 0.1) where {T} = (rng, dims...) -> fill(T(value), dims...)

vector_init(::Type{T}; value = 0.0) where {T} = (rng, dim::Integer) -> fill(T(value), dim)

function typed_inputs(::Type{T}, in_dims::Integer) where {T}
    x = T.(collect(1:in_dims)) ./ T(in_dims)
    x_batch = hcat(x, x .+ T(0.25))
    return (x, view(x_batch, :, 1), x_batch)
end

function common_model_kwargs(::Type{T}, use_bias) where {T}
    return (;
        use_bias,
        init_input = dense_init(T),
        init_reservoir = dense_init(T),
        init_bias = vector_init(T),
        init_state = dense_init(T),
    )
end

function res_model_kwargs(::Type{T}, use_bias) where {T}
    return merge(
        common_model_kwargs(T, use_bias),
        (; init_orthogonal = dense_init(T), alpha = T(0.9), beta = T(0.5)),
    )
end

function es2n_model_kwargs(::Type{T}, use_bias) where {T}
    return merge(
        common_model_kwargs(T, use_bias),
        (; init_orthogonal = dense_init(T), proximity = T(0.4)),
    )
end

function run_model_smoke(test, rng, model, ::Type{T}; batch = true) where {T}
    ps, st = setup(rng, model)
    x, x_view, x_batch = typed_inputs(T, 3)
    y, _ = model(x, ps, st)
    yv, _ = model(x_view, ps, st)
    Test.@test size(y, 1) == 2
    Test.@test size(yv, 1) == 2
    Test.@test eltype(y) <: AbstractFloat
    Test.@test eltype(yv) <: AbstractFloat
    if batch
        yb, _ = model(x_batch, ps, st)
        Test.@test size(yb) == (2, 2)
    end
    return nothing
end

end
