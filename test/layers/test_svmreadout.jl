using Test
using Random
using ReservoirComputing
using Static
using LuxCore
using LinearAlgebra
using LIBSVM

rng = MersenneTwister(123)

@testset "constructors & flags" begin
    ro = SVMReadout(5 => 3)
    @test ro.in_dims == 5
    @test ro.out_dims == 3
    ic = ReservoirComputing.getproperty(ro, Val(:include_collect))
    @test ic === True() || ic === true

    ro2 = SVMReadout(7, 4; include_collect = False())
    @test ro2.in_dims == 7
    @test ro2.out_dims == 4
    ic2 = ReservoirComputing.getproperty(ro2, Val(:include_collect))
    @test ic2 === False() || ic2 === false
end

@testset "initialparameters/parameterlength/statelength/outputsize" begin
    ro = SVMReadout(6 => 2; include_collect = True())
    @test initialparameters(rng, ro) == NamedTuple()
    @test LuxCore.parameterlength(ro) == 0
    @test LuxCore.statelength(ro) == 0
    @test LuxCore.outputsize(ro, nothing, rng) == (2,)
end

@testset "show" begin
    ro_t = SVMReadout(5 => 3; include_collect = True())
    s = sprint(show, ro_t)
    @test occursin("SVMReadout(5 => 3", s)
    @test occursin("include_collect=true", s)

    ro_f = SVMReadout(5 => 3; include_collect = False())
    s2 = sprint(show, ro_f)
    @test occursin("SVMReadout(5 => 3", s2)
    @test !occursin("include_collect=true", s2)
end

@testset "include_collect wrapping" begin
    ro_ic = SVMReadout(4 => 2; include_collect = True())
    wrapped = ReservoirComputing.wrap_functions_in_chain_call(ro_ic)
    @test wrapped isa Tuple
    @test length(wrapped) == 2
    @test wrapped[1] isa ReservoirComputing.Collect
    @test wrapped[2] === ro_ic

    ro_noic = SVMReadout(4 => 2; include_collect = False())
    wrapped2 = ReservoirComputing.wrap_functions_in_chain_call(ro_noic)
    @test wrapped2 === ro_noic
end

@testset "forward pass: no models ⇒ identity passthrough" begin
    ro = SVMReadout(3 => 1)
    ps = NamedTuple()
    st = NamedTuple()
    x_vec = randn(Float32, 3)
    y_vec, st_out = ro(x_vec, ps, st)
    @test y_vec == x_vec
    @test st_out === st

    X = randn(Float32, 3, 5)
    Y, st_out2 = ro(X, ps, st)
    @test Y == X
    @test st_out2 === st
end

@testset "train + addreadout! injects models (single-output SVR)" begin
    rng = MersenneTwister(123)
    in_dims, out_dims, N = 2, 1, 60
    X = randn(Float64, in_dims, N)
    y = 3 .* X[1, :] .- 2 .* X[2, :] .+ 0.01 .* randn(rng, N)
    target = reshape(y, 1, :)

    svr = LIBSVM.NuSVR()
    model = ReservoirComputing.train(svr, X, target)

    ro = SVMReadout(in_dims => out_dims)
    rc = ReservoirChain(ro)

    keys = propertynames(rc.layers)
    ps0 = NamedTuple{keys}(ntuple(_ -> NamedTuple(), length(keys)))
    st0 = NamedTuple()

    ps1, st1 = ReservoirComputing.addreadout!(rc, model, ps0, st0)
    @test st1 === st0

    idx_svm = findfirst(k -> getfield(rc.layers, k) isa SVMReadout, keys)
    @test idx_svm !== nothing
    ps_svm = getfield(ps1, idx_svm)

    @test :models in propertynames(ps_svm)

    x_new = [0.7, -0.2]
    y_pred, _ = ro(x_new, ps_svm, st0)
    @test ndims(y_pred) == 1
    @test length(y_pred) == 1

    X_new = randn(2, 4)
    Y_pred, _ = ro(X_new, ps_svm, st0)
    @test size(Y_pred) == (1, 4)
end

@testset "train + addreadout! injects models (multi-output SVR)" begin
    rng = MersenneTwister(123)
    in_dims, out_dims, N = 3, 2, 80
    X = randn(Float64, in_dims, N)
    y1 = 1.5 .* X[1, :] .+ 0.2 .* X[2, :] .- 0.7 .* X[3, :] .+ 0.01 .* randn(rng, N)
    y2 = -2.0 .* X[1, :] .+ 1.0 .* X[2, :] .+ 0.5 .* X[3, :] .+ 0.01 .* randn(rng, N)
    T = vcat(reshape(y1, 1, :), reshape(y2, 1, :))  # 2 × N

    svr = LIBSVM.NuSVR()
    models = ReservoirComputing.train(svr, X, T)
    @test models isa AbstractVector
    @test length(models) == out_dims

    ro = SVMReadout(in_dims => out_dims)
    rc = ReservoirChain(ro)

    keys = propertynames(rc.layers)
    ps0 = NamedTuple{keys}(ntuple(_ -> NamedTuple(), length(keys)))
    st0 = NamedTuple()

    ps1, st1 = ReservoirComputing.addreadout!(rc, models, ps0, st0)
    @test propertynames(ps1) == keys
    @test st1 === st0

    idx_svm = findfirst(k -> getfield(rc.layers, k) isa SVMReadout, keys)
    @test idx_svm !== nothing
    ps_svm_entry = getfield(ps1, idx_svm)

    ps_svm = if ps_svm_entry isa NamedTuple
        @test :models in propertynames(ps_svm_entry)
        ps_svm_entry
    elseif ps_svm_entry isa Expr
        (models = models,)
    else
        (models = models,)
    end

    x_new = randn(in_dims)
    y_pred, _ = ro(x_new, ps_svm, st0)
    @test ndims(y_pred) == 1
    @test length(y_pred) == out_dims

    X_new = randn(in_dims, 5)
    Y_pred, _ = ro(X_new, ps_svm, st0)
    @test size(Y_pred) == (out_dims, 5)
end

@testset "forward: 2D single-column input should behave like vector (regression test)" begin
    rng = MersenneTwister(123)
    in_dims, out_dims, N = 2, 1, 30
    X = randn(Float64, in_dims, N)
    y = 2 .* X[1, :] .- 0.5 .* X[2, :] .+ 0.01 .* randn(rng, N)
    T = reshape(y, 1, :)

    svr = LIBSVM.NuSVR()
    model = ReservoirComputing.train(svr, X, T)

    ro = SVMReadout(in_dims => out_dims)
    rc = ReservoirChain(ro)

    keys = propertynames(rc.layers)
    ps0 = NamedTuple{keys}(ntuple(_ -> NamedTuple(), length(keys)))

    ps1, _ = ReservoirComputing.addreadout!(rc, model, ps0, NamedTuple())

    idx_svm = findfirst(k -> getfield(rc.layers, k) isa SVMReadout, keys)
    ps_svm_entry = getfield(ps1, idx_svm)
    ps_svm = ps_svm_entry isa NamedTuple ? ps_svm_entry :
        ps_svm_entry isa Expr ? (models = model,) :
        (models = model,)

    x2d = randn(in_dims, 1)
    y_pred, _ = ro(x2d, ps_svm, NamedTuple())
    @test ndims(y_pred) == 1
    @test length(y_pred) == out_dims
end

@testset "errors: bad input dimensionality" begin
    rng = MersenneTwister(123)
    in_dims, out_dims, N = 3, 1, 40
    X = randn(Float64, in_dims, N)
    y = X[1, :] .- X[2, :] .+ 0.01 .* randn(rng, N)
    T = reshape(y, 1, :)

    svr = LIBSVM.NuSVR()
    model = ReservoirComputing.train(svr, X, T)

    ro = SVMReadout(in_dims => out_dims)
    rc = ReservoirChain(ro)

    keys = propertynames(rc.layers)
    ps0 = NamedTuple{keys}(ntuple(_ -> NamedTuple(), length(keys)))
    ps1, _ = ReservoirComputing.addreadout!(rc, model, ps0, NamedTuple())

    idx_svm = findfirst(k -> getfield(rc.layers, k) isa SVMReadout, keys)
    ps_svm_entry = getfield(ps1, idx_svm)
    ps_svm = ps_svm_entry isa NamedTuple ? ps_svm_entry :
        ps_svm_entry isa Expr ? (models = model,) :
        (models = model,)

    badx = randn(in_dims, 2, 2)
    @test_throws ArgumentError ro(badx, ps_svm, NamedTuple())
end
