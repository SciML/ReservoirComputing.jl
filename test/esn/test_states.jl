using ReservoirComputing, Random

const res_size = 20
const ts = 0.:0.1:50.0
const data = sin.(ts)
const train_len = 400
const input_data = reduce(hcat, data[1:train_len-1])
const target_data = reduce(hcat, data[2:train_len])
const predict_len = 100
const test_data = reduce(hcat, data[train_len+1:train_len+predict_len])

states_types = [StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates]

for t in states_types
    Random.seed!(77)
    esn = ESN(res_size, input_data;
        reservoir=RandSparseReservoir(1.2, 0.1))
    output_layer = train(esn, target_data)
    output = esn(Generative(predict_len), output_layer)
    @test maximum(abs.(test_data .- output)) ./ maximum(abs.(test_data)) < 0.1
end
