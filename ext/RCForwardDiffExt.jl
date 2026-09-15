module RCForwardDiffExt

using ReservoirComputing: ReservoirComputing
using ForwardDiff: ForwardDiff

function forwarddiff_closedloop_jacobian!(
        J::AbstractMatrix, esn::ReservoirComputing.ESN, x::AbstractVector, ps
    )
    ReservoirComputing.__require_esn_closedloop_io(esn)
    closed_loop = let esn = esn, ps = ps
        state -> first(ReservoirComputing.__closed_loop_step(esn, state, ps))
    end
    J .= ForwardDiff.jacobian(closed_loop, x)
    return J
end

end # module
