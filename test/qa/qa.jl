using SciMLTesting, ReservoirComputing, Test
using JET

# ExplicitImports only checks an extension module once it exists, and an extension only
# exists once its trigger package is loaded. Loading the weakdeps here is what puts
# RCCellularAutomataExt, RCODEReservoirExt, RCLIBSVMExt, RCMLJLinearModelsExt,
# RCSparseArraysExt and RCStateSpaceSetsExt in scope for the QA checks.
using CellularAutomata, DataInterpolations, LIBSVM, MLJLinearModels, SparseArrays,
    StateSpaceSets

# ReservoirComputing's own extension hook points. ExplicitImports' `allow_internal_imports`
# / `allow_internal_accesses` defaults would cover these, but they key off
# `Base.moduleroot`, and an extension is its own root module rather than a submodule of
# the package it extends -- so a package reaching into its *own* internals from its *own*
# extension reads as an external non-public access.
rc_internal_hooks = (
    :AbstractInputEncoding,
    :AbstractReservoirComputer,
    :IntegerType,
    :__apply_seq,
    :__check_lsm_components,
    :__check_lsm_tspan,
    :__check_protected_kwargs,
    :__collectstates,
    :__continuous_esn_rhs!,
    :__feature_dim,
    :__fit_readout,
    :__init_encoder_st,
    :__predict,
    :__reservoir_jac_prototype,
    :__supports_ar,
    :__wrap_layers,
    :addreadout!,
)

# The LuxCore and WeightInitializers API that ReservoirComputing deliberately reexports so
# that `using ReservoirComputing` on its own is enough to follow the tutorials. Owned and
# documented upstream; kept in sync with the reexport `export` block in
# src/ReservoirComputing.jl.
const REEXPORTS = (
    :apply, :initialparameters, :initialstates, :setup,
    :orthogonal, :rand32, :randn32, :sparse_init, :zeros32,
)

run_qa(
    ReservoirComputing;
    reexports_allow = REEXPORTS,
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                rc_internal_hooks...,
                # LinearAlgebra doesn't declare its SVD algorithm selectors public.
                :QRIteration,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                rc_internal_hooks...,
                # LIBSVM neither exports nor declares `AbstractSVR` public, and it is the
                # only supertype covering all of its regression models -- `__fit_readout`
                # has to dispatch on it.
                :AbstractSVR,
                # `Base.@deprecate_binding` is Base's own standard deprecation mechanism
                # but isn't declared `public` in Base.
                :var"@deprecate_binding",
            ),
        ),
    )
)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using ReservoirComputing`,
    # so the allow-list cannot drift into approving names the package no longer provides.
    # `isdefined(@__MODULE__, ...)` tests the property directly: this file's `using
    # ReservoirComputing` is what has to bring the name into scope.
    @testset "$name" for name in REEXPORTS
        @test name in names(ReservoirComputing)
        @test isdefined(@__MODULE__, name)
    end
end
