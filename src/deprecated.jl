# Remove this file and test/deprecated.jl at v1.0.
# https://invenia.github.io/blog/2022/06/17/deprecating-in-julia/

Base.@deprecate_binding StandardRidge RidgeRegression

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
