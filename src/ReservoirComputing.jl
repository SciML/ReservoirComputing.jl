module ReservoirComputing

using ArrayInterface: ArrayInterface
using ConcreteStructs: @concrete
using LinearAlgebra: eigvals, I, qr, Diagonal, diag, mul!
using LuxCore: AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer,
    setup, apply, replicate
import LuxCore: initialparameters, initialstates, statelength, outputsize
using NNlib: tanh_fast
using Random: Random, AbstractRNG, randperm
using Static: StaticBool, StaticSymbol, True, False, static, known, StaticInteger
using Reexport: Reexport, @reexport
using WeightInitializers: WeightInitializers, DeviceAgnostic, PartialFunction, Utils,
    orthogonal, rand32, randn32, sparse_init, zeros32
@reexport using WeightInitializers
@reexport using LuxCore: setup, apply, initialparameters, initialstates

#@compat(public, (initialparameters)) #do I need to add intialstates/parameters in compat?

#reservoir computers
include("generics.jl")
include("reservoircomputer.jl")
#layers
include("layers/basic.jl")
include("layers/lux_layers.jl")
include("layers/additive_eiesn_cell.jl")
include("layers/eiesn_cell.jl")
include("layers/es2n_cell.jl")
include("layers/esn_cell.jl")
include("layers/eusn_cell.jl")
include("layers/svmreadout.jl")
#general
include("states.jl")
include("predict.jl")
include("train.jl")
#initializers
include("inits/inits_components.jl")
include("inits/inits_input.jl")
include("inits/inits_reservoir.jl")
#full models
include("models/esn_generics.jl")
include("models/esn.jl")
include("models/additive_eiesn.jl")
include("models/eiesn.jl")
include("models/es2n.jl")
include("models/esn_deep.jl")
include("models/esn_delay.jl")
include("models/esn_hybrid.jl")
include("models/eusn.jl")
include("models/ngrc.jl")
#extensions
include("extensions/reca.jl")

export ReservoirComputer
export AdditiveEIESNCell, EIESNCell, ES2NCell, ESNCell, EuSNCell
export Collect, collectstates, DelayLayer, LinearReadout, NonlinearFeaturesLayer,
    ReservoirChain, StatefulLayer
export SVMReadout
export Extend, ExtendedSquare, NLAT1, NLAT2, NLAT3, Pad, PartialSquare
export StandardRidge
export chebyshev_mapping, informed_init, logistic_mapping, minimal_init,
    modified_lm, scaled_rand, weighted_init, weighted_minimal
export block_diagonal, chaotic_init, cycle_jumps, delay_line, delayline_backward,
    diagonal_init, double_cycle, forward_connection, low_connectivity, permutation_init,
    pseudo_svd, rand_sparse, selfloop_backward_cycle, selfloop_cycle, selfloop_delayline_backward,
    selfloop_forwardconnection, simple_cycle, true_doublecycle
export add_jumps!, backward_connection!, delay_line!, permute_matrix!, reverse_simple_cycle!,
    scale_radius!, self_loop!, simple_cycle!
export polynomial_monomials, predict, QRSolver, resetcarry!, train, train!
export AdditiveEIESN, DeepESN, DelayESN, EIESN, ES2N, ESN, EuSN, HybridESN, InputDelayESN, StateDelayESN
export NGRC
#ext
export RECACell, RECA
export RandomMapping, RandomMaps

#precompilation
include("precompilation.jl")

end #module
