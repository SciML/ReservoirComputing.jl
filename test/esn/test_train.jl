using ReservoirComputing, MLJLinearModels, Random, Statistics, LIBSVM

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
esn = ESN(input_data, 1, res_size;
    reservoir = rand_sparse)

training_methods = [
    StandardRidge(regularization_coeff = reg),
    LinearModel(RidgeRegression, regression_kwargs = (; lambda = reg)),
    LinearModel(regression = RidgeRegression, regression_kwargs = (; lambda = reg)),
    EpsilonSVR()
]

# TODO check types
@testset "Training Algo Tests: $ta" for ta in training_methods
    output_layer = train(esn, target_data, ta)
    output = esn(Predictive(input_data), output_layer)
    @test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.22
end
