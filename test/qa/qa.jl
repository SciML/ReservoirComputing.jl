using SciMLTesting, ReservoirComputing
using JET

# ExplicitImports only checks an extension module once it exists, and an extension only
# exists once its trigger package is loaded. Loading the weakdeps here is what puts
# RCCellularAutomataExt, RCODEReservoirExt, RCLIBSVMExt, RCMLJLinearModelsExt and
# RCSparseArraysExt in scope for the QA checks.
using CellularAutomata, DataInterpolations, LIBSVM, MLJLinearModels, SparseArrays

# ReservoirComputing's own extension hook points. ExplicitImports' `allow_internal_imports`
# / `allow_internal_accesses` defaults would cover these, but they key off
# `Base.moduleroot`, and an extension is its own root module rather than a submodule of
# the package it extends -- so a package reaching into its *own* internals from its *own*
# extension reads as an external non-public access.
rc_internal_hooks = (
    :AbstractInputEncoding,
    :AbstractReservoirComputer,
    :IntegerType,
    :_apply_seq,
    :_check_protected_kwargs,
    :_collectstates,
    :_continuous_esn_rhs!,
    :__check_lsm_tspan,
    :__feature_dim,
    :__init_encoder_st,
    :__supports_ar,
    :_fit_readout,
    :_predict,
    :_reservoir_jac_prototype,
    :_wrap_layers,
    :addreadout!,
)

run_qa(
    ReservoirComputing;
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = rc_internal_hooks),
        all_qualified_accesses_are_public = (;
            ignore = (
                rc_internal_hooks...,
                # LIBSVM neither exports nor declares `AbstractSVR` public, and it is the
                # only supertype covering all of its regression models -- `_fit_readout`
                # has to dispatch on it.
                :AbstractSVR,
            ),
        ),
    )
)
