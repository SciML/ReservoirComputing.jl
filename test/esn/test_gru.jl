using ReservoirComputing, Random, Statistics

const res_size=20
const ts = 0.:0.1:50.0
const data = sin.(ts)
const train_len = 400
const input_data = reduce(hcat, data[1:train_len-1])
const target_data = reduce(hcat, data[2:train_len])
const predict_len = 100
const test = reduce(hcat, data[train_len+1:train_len+predict_len])

variants = [FullyGated(), Variant1(), Variant2(), Variant3(), Minimal()]
for v in variants
    Random.seed!(77)
    esn = ESN(res_size, input_data; 
        reservoir_init=RandSparseReservoir(1.2, 0.1),
        reservoir_driver = GRU(variant=v, reservoir_init=[RandSparseReservoir(1., 0.5), RandSparseReservoir(1.2, 0.1)]))
    
    output_layer = train(esn, target_data)
    output = esn(Predictive(target_data), output_layer, initial_conditions=target_data[1])
    @test mean(abs.(target_data .- output)) ./ mean(abs.(target_data)) < 0.11
end