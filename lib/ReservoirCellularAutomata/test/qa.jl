using ReservoirCellularAutomata, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(ReservoirCellularAutomata)
    Aqua.test_ambiguities(ReservoirCellularAutomata; recursive = false)
    Aqua.test_deps_compat(ReservoirCellularAutomata)
    Aqua.test_piracies(ReservoirCellularAutomata)
    Aqua.test_project_extras(ReservoirCellularAutomata)
    Aqua.test_stale_deps(ReservoirCellularAutomata)
    Aqua.test_unbound_args(ReservoirCellularAutomata)
    Aqua.test_undefined_exports(ReservoirCellularAutomata)
end
