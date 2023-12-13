using ReservoirComputing, Random

const res_size = 20
const ts = 0.0:0.1:50.0
const data = sin.(ts)
const train_len = 400
const input_data = reduce(hcat, data[1:(train_len - 1)])
const target_data = reduce(hcat, data[2:train_len])
const predict_len = 100
const test = reduce(hcat, data[(train_len + 1):(train_len + predict_len)])
const training_method = StandardRidge(10e-6)

nlas = [NLADefault(), NLAT1(), NLAT2(), NLAT3()]

for n in nlas
    Random.seed!(77)
    esn = ESN(input_data;
        reservoir = RandSparseReservoir(res_size, 1.2, 0.1),
        nla_type = n)
    output_layer = train(esn, target_data, training_method)
    output = esn(Generative(predict_len), output_layer)
    @test maximum(abs.(test .- output)) ./ maximum(abs.(test)) < 0.1
end
