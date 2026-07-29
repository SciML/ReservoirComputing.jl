# Remove this file and test/deprecated.jl at v1.0.
# https://invenia.github.io/blog/2022/06/17/deprecating-in-julia/

Base.@deprecate_binding StandardRidge RidgeRegression

export toepliz_init

@doc raw"""
    toepliz_init([rng], [T], dims...; kwargs...)

!!! warning "Deprecated"
    `toepliz_init` was a misspelling of `toeplitz_init`. Use
    [`toeplitz_init`](@ref) instead.
"""
function toepliz_init(args...; kwargs...)
    Base.depwarn(
        "`toepliz_init` is deprecated; use `toeplitz_init` instead.",
        :toepliz_init
    )
    return toeplitz_init(args...; kwargs...)
end

@doc raw"""
    train!(rc, train_data, target_data, ps, st,
           train_method=RidgeRegression(0.0);
           washout=0, return_states=false, kwargs...)

!!! warning "Deprecated"
    Use [`train`](@ref) instead. The positional `train_method` argument maps
    to the `objective` keyword of `train`.
"""
function train!(
        rc, train_data, target_data, ps, st,
        train_method = RidgeRegression(0.0);
        washout::Integer = 0, return_states::Bool = false, kwargs...
    )
    Base.depwarn(
        "`train!` is deprecated; use `train(rc, train_data, target_data, ps, st; " *
            "objective=..., solver=..., washout=..., return_states=...)` instead.",
        :train!
    )
    return train(
        rc, train_data, target_data, ps, st;
        objective = train_method,
        washout = washout,
        return_states = return_states,
        kwargs...
    )
end

@doc raw"""
    train(objective, states, target_data; solver=nothing, kwargs...)

!!! warning "Deprecated"
    The objective-level entry point is deprecated. Use the model-level
    `train(rc, train_data, target_data, ps, st; objective=..., solver=...)`
    API instead.
"""
function train(
        objective, states::AbstractMatrix, target_data::AbstractMatrix;
        solver = nothing, kwargs...
    )
    Base.depwarn(
        "`train(objective, states, target_data; solver, kwargs...)` is deprecated. " *
            "Use the model-level `train(rc, train_data, target_data, ps, st; " *
            "objective=..., solver=...)` API instead.",
        :train
    )
    if isnothing(solver)
        return _fit_readout(objective, states, target_data; kwargs...)
    end
    return _fit_readout(objective, states, target_data; solver = solver, kwargs...)
end
