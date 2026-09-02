module RCStateSpaceSetsExt

using ReservoirComputing: ReservoirComputing
using StateSpaceSets: AbstractStateSpaceSet, StateSpaceSet

function ReservoirComputing.train(
        rc, train_data::AbstractStateSpaceSet, target_data::AbstractStateSpaceSet, ps, st;
        kwargs...
    )
    return ReservoirComputing.train(
        rc, __as_rc_matrix(train_data), __as_rc_matrix(target_data), ps, st; kwargs...
    )
end

function ReservoirComputing.predict(rc, data::AbstractStateSpaceSet, ps, st)
    outputs, st = ReservoirComputing.predict(rc, __as_rc_matrix(data), ps, st)
    return __as_statespaceset(outputs), st
end

function __as_rc_matrix(set::AbstractStateSpaceSet{D, T}) where {D, T}
    isempty(set) && return Matrix{T}(undef, D, 0)
    return stack(set)
end

function __as_statespaceset(data::AbstractMatrix)
    return StateSpaceSet(collect(eachcol(data)))
end

end #module
