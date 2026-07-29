using SciMLTesting, ReservoirComputing, Test
using JET

run_qa(
    ReservoirComputing;
    explicit_imports = true,
    reexports_allow = (:QRFactorization,),  # QRFactorization aliases LinearSolve's
    jet_kwargs = (;
        target_modules = (ReservoirComputing,),
        mode = :typo,
        toplevel_logger = nothing,
    ),
    api_docs_kwargs = (;
        rendered = true,
        rendered_ignore = (
            :StandardRidge,  # deprecated alias of RidgeRegression
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :OneTo,        # Base (non-public)
                :tail,         # Base (non-public)
                Symbol("@deprecate_binding"),  # Base (non-public deprecation macro)
                :aos_to_soa,   # ArrayInterface (non-public)
                :Partial,      # WeightInitializers.PartialFunction (non-public)
                :default_rng,  # WeightInitializers.Utils (non-public)
                :ones,         # WeightInitializers.DeviceAgnostic (non-public)
                :rand,         # WeightInitializers.DeviceAgnostic (non-public)
                :zeros,        # WeightInitializers.DeviceAgnostic (non-public)
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :return_init_as,     # ReservoirComputing (own non-public name)
                :DeviceAgnostic,     # WeightInitializers (non-public)
                :PartialFunction,    # WeightInitializers (non-public)
                :Utils,              # WeightInitializers (non-public)
                :StaticInteger,      # Static (non-public)
                :apply,              # LuxCore (non-public)
                :initialparameters,  # LuxCore (non-public)
                :initialstates,      # LuxCore (non-public)
                :outputsize,         # LuxCore (non-public)
                :replicate,          # LuxCore (non-public)
                :setup,              # LuxCore (non-public)
                :statelength,        # LuxCore (non-public)
                :SciMLLinearSolveAlgorithm,  # LinearSolve (non-public abstract type)
                :needs_square_A,     # LinearSolve (non-public)
            ),
        ),
    )
)
