module ReservoirComputing

using ArrayInterface: ArrayInterface
using ConcreteStructs: @concrete
using LinearAlgebra: eigvals, eigen, I, qr, Diagonal, diag, mul!, Symmetric, norm,
    svd, svdvals, nullspace, pinv, tr, checksquare, issymmetric, LAPACKException,
    QRIteration
using LinearSolve: LinearProblem, solve
using LinearSolve: QRFactorization as LinearSolveQRFactorization
using LuxCore: AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer,
    setup, apply, replicate
import LuxCore: initialparameters, initialstates, statelength, outputsize
using NNlib: tanh_fast
using Random: Random, AbstractRNG, randperm
using SciMLBase: AbstractLinearAlgorithm, successful_retcode
using Static: StaticBool, StaticSymbol, StaticInt, True, False, static, known
using WeightInitializers: orthogonal, rand32, randn32, sparse_init, zeros32

include("initializer_utils.jl")

#@compat(public, (initialparameters)) #do I need to add intialstates/parameters in compat?

#reservoir computers
include("generics.jl")
include("layers/sciml_reservoir.jl")
include("reservoircomputer.jl")
#layers
include("layers/basic.jl")
include("layers/lux_layers.jl")
include("layers/esn_cell.jl")
include("layers/continuous_esn_cell.jl")
include("layers/additive_eiesn_cell.jl")
include("layers/eiesn_cell.jl")
include("layers/es2n_cell.jl")
include("layers/resesn_cell.jl")
include("layers/eusn_cell.jl")
include("layers/memoryesn_cell.jl")
include("layers/memoryresesn_cell.jl")
include("layers/rmn_cell.jl")
include("layers/svmreadout.jl")
include("layers/lif_wrapper.jl")
#general
include("states.jl")
include("predict.jl")
include("train.jl")
#initializers
include("inits/inits_components.jl")
include("inits/inits_input.jl")
include("inits/inits_reservoir.jl")
include("inits/inits_lsm.jl")
include("layers/spiking.jl")
include("deprecated.jl")
#full models
include("models/esn_generics.jl")
include("models/esn.jl")
include("models/additive_eiesn.jl")
include("models/eiesn.jl")
include("models/es2n.jl")
include("models/resesn.jl")
include("models/esn_deep.jl")
include("models/esn_delay.jl")
include("models/esn_hybrid.jl")
include("models/eusn.jl")
include("models/svesm.jl")
include("models/lifesn.jl")
include("models/ngrc.jl")
include("models/rmnesn.jl")
include("models/rmnresesn.jl")
include("models/continuous_esn.jl")
include("models/lsm.jl")
#conceptors
include("conceptors.jl")
#extensions
include("extensions/reca.jl")

export ReservoirComputer
export AbstractSciMLProblemReservoir, SciMLProblemReservoir, ContinuousESN, LSM
export AbstractSampler, TerminalStateSampling
export ContinuousESNCell, LSMCell
export AbstractSpikingNeuron, LIFNeuron
export AbstractInputEncoder, CurrentInjection, PoissonRateEncoder
export AbstractSpikeFeature, SpikeCountFeatures, ExponentialSpikeFilter,
    MembraneVoltageFeature
export AdditiveEIESNCell, EIESNCell, ES2NCell, ESNCell, EuSNCell, LIFESNCell,
    MemoryESNCell, MemoryResESNCell, ResESNCell, RMNCell
export StatefulLayer, LinearReadout, ReservoirChain, Collect, collectstates,
    DelayLayer, NonlinearFeaturesLayer
export SVMReadout
export LocalInformationFlow
export Extend, ExtendedSquare, NLAT1, NLAT2, NLAT3, Pad, PartialSquare
export RidgeRegression, StandardRidge
export chebyshev_mapping, informed_init, logistic_mapping, minimal_init,
    modified_lm, scaled_rand, weighted_init, weighted_minimal
export band_init, block_diagonal, chaotic_init, cycle_jumps, delay_line, delayline_backward,
    diagonal_init, double_cycle, forward_connection, low_connectivity, lower_triangular, permutation_init,
    pseudo_svd, rand_hyper, rand_sparse, dale_sparse, selfloop_backward_cycle, selfloop_cycle, selfloop_delayline_backward,
    selfloop_forwardconnection, simple_cycle, toeplitz_init, true_doublecycle, wigner_init
export add_jumps!, backward_connection!, delay_line!, permute_matrix!, reverse_simple_cycle!,
    scale_radius!, self_loop!, simple_cycle!
export polynomial_monomials, chebyshev_monomials, predict, QRSolver, QRFactorization,
    resetcarry!, return_init_as, train, train!
export AdditiveEIESN, DeepESN, DelayESN, EIESN, ES2N, ESN, EuSN, HybridESN, InputDelayESN, LIFESN, ResESN, StateDelayESN, SVESM
export NGRC
export RMNESN, RMNResESN
#conceptors
export Conceptor, correlation_matrix, conceptor_matrix, conceptor_from_states,
    conceptor_singular_values, quota
export aperture_adapt, adapt_singular_value, reaperture, attenuation, optimal_aperture
export conceptor_not, conceptor_and, conceptor_or
export has_conceptor, get_conceptor, store_conceptor, store_conceptors,
    set_active_conceptor, active_conceptor
export loadpatterns, generate, morph_conceptor
#ext
export RECACell, RECA
export RandomMapping, RandomMaps

#precompilation
include("precompilation.jl")

end #module
