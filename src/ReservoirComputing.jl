module ReservoirComputing

using ArrayInterface: ArrayInterface
using Compat: @compat
using ConcreteStructs: @concrete
using LinearAlgebra: eigvals, mul!, I, qr, Diagonal, diag
using LuxCore: AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer,
               setup, apply, replicate
import LuxCore: initialparameters, initialstates, statelength, outputsize
using NNlib: fast_act, sigmoid
using Random: Random, AbstractRNG, randperm
using Static: StaticBool, StaticInt, StaticSymbol,
              True, False, static, known, dynamic, StaticInteger
using Reexport: Reexport, @reexport
using WeightInitializers: DeviceAgnostic, PartialFunction, Utils
@reexport using WeightInitializers
@reexport using LuxCore: setup, apply, initialparameters, initialstates

#@compat(public, (initialparameters)) #do I need to add intialstates/parameters in compat?

#reservoir computers
include("generics.jl")
include("reservoircomputer.jl")
#layers
include("layers/basic.jl")
include("layers/lux_layers.jl")
include("layers/esn_cell.jl")
include("layers/es2n_cell.jl")
include("layers/eusn_cell.jl")
include("layers/paesn_cell.jl")
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
include("models/es2n.jl")
include("models/esn.jl")
include("models/esn_deep.jl")
include("models/esn_delay.jl")
include("models/esn_hybrid.jl")
include("models/eusn.jl")
include("models/paesn.jl")
include("models/ngrc.jl")
#extensions
include("extensions/reca.jl")

export ReservoirComputer
export ESNCell, ES2NCell, EuSNCell, ParameterAwareESNCell, ParameterAwareStatefulLayer
export StatefulLayer, LinearReadout, ReservoirChain, Collect, collectstates,
       DelayLayer, NonlinearFeaturesLayer
export SVMReadout
export Pad, Extend, NLAT1, NLAT2, NLAT3, PartialSquare, ExtendedSquare
export StandardRidge
export chebyshev_mapping, informed_init, logistic_mapping, minimal_init,
       modified_lm, scaled_rand, weighted_init, weighted_minimal
export block_diagonal, chaotic_init, cycle_jumps, delay_line, delayline_backward,
       double_cycle, forward_connection, low_connectivity, pseudo_svd, rand_sparse,
       selfloop_cycle, selfloop_delayline_backward, selfloop_backward_cycle,
       selfloop_forwardconnection, simple_cycle, true_doublecycle
export add_jumps!, backward_connection!, delay_line!, reverse_simple_cycle!,
       scale_radius!, self_loop!, simple_cycle!
export train, train!, predict, resetcarry!, polynomial_monomials
export ES2N, ESN, EuSN, DeepESN, DelayESN, HybridESN, ParameterAwareESN
export NGRC
#ext
export RECACell, RECA
export RandomMapping, RandomMaps

end #module
