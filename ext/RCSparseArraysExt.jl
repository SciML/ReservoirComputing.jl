module RCSparseArraysExt
import ReservoirComputing
using SparseArrays: sparse

function ReservoirComputing.return_init_as(::Val{true}, layer_matrix::AbstractVecOrMat)
    return sparse(layer_matrix)
end

end #module
