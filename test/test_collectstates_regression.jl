using Test
using Random
using ReservoirComputing
using LinearAlgebra
using Static

const SEED = 42
const IN_DIMS = 3
const RES_DIMS = 8
const OUT_DIMS = 2
const T = 20

_seeded_input(rng, m, n) = (Random.seed!(MersenneTwister(101), 101);
    0.5f0 .* randn(MersenneTwister(101), Float32, m, n))
_seeded_reservoir(rng, m, n) = 0.3f0 .* randn(MersenneTwister(202), Float32, m, n)
_seeded_bias(rng, m) = 0.1f0 .* randn(MersenneTwister(303), Float32, m)
init_state3(rng, m, B) = B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)

fixed_kwargs() = (
    use_bias = False(),
    init_input = _seeded_input,
    init_reservoir = _seeded_reservoir,
    init_bias = _seeded_bias,
    init_state = init_state3,
)

function build_models()
    fk = fixed_kwargs()
    return [
        ("ESN",
            ESN(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk..., leak_coefficient = 1.0)),
        ("EuSN",
            EuSN(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk..., leak_coefficient = 1.0)),
        ("ES2N",
            ES2N(IN_DIMS, RES_DIMS, OUT_DIMS, identity;
                fk..., proximity = 1.0, init_orthogonal = _seeded_reservoir)),
        ("ResESN",
            ResESN(IN_DIMS, RES_DIMS, OUT_DIMS, identity;
                fk..., beta = 1.0, init_orthogonal = _seeded_reservoir, alpha = 1.0)),
        ("SVESM",
            SVESM(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk...)),
        ("LIFESN",
            LIFESN(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk...)),
    ]
end

function collect_hash(model)
    rng = MersenneTwister(SEED)
    ps, st = setup(rng, model)
    data = randn(MersenneTwister(SEED + 1), Float32, IN_DIMS, T)
    states, _ = collectstates(model, data, ps, st)
    return hash(states), size(states)
end

@testset "collectstates dispatch regression" begin
    expected = Dict{String, Tuple{UInt64, NTuple{2, Int}}}(
        "ESN"    => (0xdb5186679ae62882, (8, 20)),
        "EuSN"   => (0x4e19abcd57a3f3cc, (8, 20)),
        "ES2N"   => (0xdb5186679ae62882, (8, 20)),
        "ResESN" => (0x59583b44574cf3d8, (8, 20)),
        "SVESM"  => (0xdb5186679ae62882, (8, 20)),
        "LIFESN" => (0x6b1ca505dc6cb1e8, (8, 20)),
    )

    for (name, model) in build_models()
        @testset "$name" begin
            h, sz = collect_hash(model)
            if isempty(expected)
                @info "FIXTURE" name h sz
            else
                ref_h, ref_sz = expected[name]
                @test sz == ref_sz
                @test h == ref_h
            end
        end
    end
end
