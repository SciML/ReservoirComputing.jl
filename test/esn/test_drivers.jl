using ReservoirComputing, Random, Statistics, NNlib

const res_size = 50
const ts = 0.0:0.1:50.0
const data = sin.(ts)
const train_len = 400
const input_data = reduce(hcat, data[1:(train_len - 1)])
const target_data = reduce(hcat, data[2:train_len])
const predict_len = 100
const test_data = reduce(hcat, data[(train_len + 1):(train_len + predict_len)])
const training_method = StandardRidge(10e-6)
Random.seed!(77)

function test_esn(input_data, target_data, training_method, esn_config)
    esn = ESN(input_data, 1, res_size; esn_config...)
    output_layer = train(esn, target_data, training_method)

    output = esn(Predictive(target_data), output_layer; initial_conditions = target_data[1])
    @test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.15
end

esn_configs = [
    Dict(:reservoir => rand_sparse(; radius = 1.2),
        :reservoir_driver => GRU(; variant = FullyGated(),
            reservoir = [
                rand_sparse(; radius = 1.0, sparsity = 0.5),
                rand_sparse(; radius = 1.2, sparsity = 0.1)
            ])),
    Dict(:reservoir => rand_sparse(; radius = 1.2),
        :reservoir_driver => GRU(; variant = Minimal(),
            reservoir = rand_sparse(; radius = 1.0, sparsity = 0.5),
            inner_layer = scaled_rand,
            bias = scaled_rand)),
    Dict(:reservoir => rand_sparse(; radius = 1.2),
        :reservoir_driver => MRNN(; activation_function = (tanh, sigmoid),
            scaling_factor = (0.8, 0.1)))
]

@testset "Test Drivers: $config" for config in esn_configs
    test_esn(input_data, target_data, training_method, config)
end
