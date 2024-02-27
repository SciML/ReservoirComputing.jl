using ReservoirComputing, Random, Statistics

const res_size = 20
const ts = 0.0:0.1:50.0
const data = sin.(ts)
const train_len = 400
const predict_len = 100
const input_data = reduce(hcat, data[1:(train_len - 1)])
const target_data = reduce(hcat, data[2:train_len])
const test = reduce(hcat, data[(train_len + 1):(train_len + predict_len)])
const reg = 10e-6
#test_types = [Float64, Float32, Float16]

Random.seed!(77)
res = rand_sparse(; radius = 1.2, sparsity = 0.1)
esn = DeepESN(input_data, 1, res_size)

output_layer = train(esn, target_data)
output = esn(Generative(length(test)), output_layer)
@test mean(abs.(test .- output)) ./ mean(abs.(test)) < 0.22
