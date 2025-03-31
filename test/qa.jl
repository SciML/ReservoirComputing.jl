using ReservoirComputing, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(ReservoirComputing)
    Aqua.test_ambiguities(ReservoirComputing; recursive = false)
    Aqua.test_deps_compat(ReservoirComputing)
    Aqua.test_piracies(ReservoirComputing)
    Aqua.test_project_extras(ReservoirComputing)
    Aqua.test_stale_deps(ReservoirComputing)
    Aqua.test_unbound_args(ReservoirComputing)
    Aqua.test_undefined_exports(ReservoirComputing)
end
