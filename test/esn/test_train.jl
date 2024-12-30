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
    reservoir = res)
# different models that implement a train dispatch
# TODO add classification
linear_training = [StandardRidge(0.0), LinearRegression(; fit_intercept = false),
    RidgeRegression(; fit_intercept = false), LassoRegression(; fit_intercept = false),
    ElasticNetRegression(; fit_intercept = false), HuberRegression(; fit_intercept = false),
    QuantileRegression(; fit_intercept = false), LADRegression(; fit_intercept = false)]
svm_training = [EpsilonSVR(), NuSVR()]

# TODO check types
@testset "Linear training: $lt" for lt in linear_training
    output_layer = train(esn, target_data, lt)
    @test output_layer isa OutputLayer
    @test output_layer.output_matrix isa AbstractArray
end

@testset "SVM training: $st" for st in svm_training
    output_layer = train(esn, target_data, st)
    @test output_layer isa OutputLayer
    @test output_layer.output_matrix isa typeof(st)
end
