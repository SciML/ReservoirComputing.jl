module RCSparseArraysExt
import ReservoirComputing: return_init_as
using SparseArrays: sparse

function return_init_as(::Val{true}, layer_matrix::AbstractVecOrMat)
    return sparse(layer_matrix)
end

end #module
