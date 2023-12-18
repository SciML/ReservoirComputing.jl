using ReservoirComputing, Random, Statistics, NNlib

const res_size = 20
const ts = 0.0:0.1:50.0
const data = sin.(ts)
const train_len = 400
const input_data = reduce(hcat, data[1:(train_len - 1)])
const target_data = reduce(hcat, data[2:train_len])
const predict_len = 100
const test = reduce(hcat, data[(train_len + 1):(train_len + predict_len)])
const training_method = StandardRidge(10e-6)

Random.seed!(77)
esn = ESN(input_data;
    reservoir = RandSparseReservoir(res_size, 1.2, 0.1),
    reservoir_driver = GRU(variant = FullyGated(),
        reservoir = [
            RandSparseReservoir(res_size, 1.0, 0.5),
            RandSparseReservoir(res_size, 1.2, 0.1),
        ]))

output_layer = train(esn, target_data, training_method)
output = esn(Predictive(target_data), output_layer, initial_conditions = target_data[1])
@test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.11

esn = ESN(input_data;
    reservoir = RandSparseReservoir(res_size, 1.2, 0.1),
    reservoir_driver = GRU(variant = Minimal(),
        reservoir = RandSparseReservoir(res_size, 1.0, 0.5),
        inner_layer = DenseLayer(),
        bias = DenseLayer()))

output_layer = train(esn, target_data, training_method)
output = esn(Predictive(target_data), output_layer, initial_conditions = target_data[1])
@test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.11

#multiple rnn
esn = ESN(input_data;
    reservoir = RandSparseReservoir(res_size, 1.2, 0.1),
    reservoir_driver = MRNN(activation_function = (tanh, sigmoid),
        scaling_factor = (0.8, 0.1)))
output_layer = train(esn, target_data, training_method)
output = esn(Predictive(target_data), output_layer, initial_conditions = target_data[1])
@test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.11

#deep esn
esn = ESN(input_data;
    reservoir = [
        RandSparseReservoir(res_size, 1.2, 0.1),
        RandSparseReservoir(res_size, 1.2, 0.1),
    ])
output_layer = train(esn, target_data, training_method)
output = esn(Predictive(target_data), output_layer, initial_conditions = target_data[1])
@test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.11

esn = ESN(input_data;
    reservoir = [
        RandSparseReservoir(res_size, 1.2, 0.1),
        RandSparseReservoir(res_size, 1.2, 0.1),
    ],
    input_layer = [DenseLayer(), DenseLayer()],
    bias = [NullLayer(), NullLayer()])
output_layer = train(esn, target_data, training_method)
output = esn(Predictive(target_data), output_layer, initial_conditions = target_data[1])
@test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.11
