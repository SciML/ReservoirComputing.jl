using SciMLTesting, ReservoirComputing, Test
using JET

run_qa(
    ReservoirComputing;
    explicit_imports = true,
    jet_kwargs = (;
        target_modules = (ReservoirComputing,),
        mode = :typo,
        toplevel_logger = nothing,
    ),
    api_docs_kwargs = (;
        rendered = true,
        rendered_ignore = (
            :WeightInitializers, :apply, :glorot_normal, :glorot_uniform, :identity_init,
            :initialparameters, :initialstates, :kaiming_normal, :kaiming_uniform,
            :ones16, :ones32, :ones64, :onesC16, :onesC32, :onesC64, :orthogonal,
            :rand16, :rand32, :rand64, :randC16, :randC32, :randC64,
            :randn16, :randn32, :randn64, :randnC16, :randnC32, :randnC64,
            :setup, :sparse_init, :truncated_normal, :zeros16, :zeros32, :zeros64,
            :zerosC16, :zerosC32, :zerosC64,
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :OneTo,        # Base (non-public)
                :tail,         # Base (non-public)
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
            ),
        ),
    )
)
