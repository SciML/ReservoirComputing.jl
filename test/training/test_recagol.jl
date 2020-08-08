using ReservoirComputing

train_data = ones(Int, 2, 10)
res_size = 10
generations = 5
permutations = 2

reca = RECA_TwoDim(train_data, res_size, generations, permutations)

W_out = ESNtrain(reca, 0.001, train_data = ones(Float64, 2, 10))
output = RECATD_predict_discrete(reca, 10, W_out)

@test isequal(size(output), size(ones(Int, 2, 10)))

output = RECATDdirect_predict_discrete(reca, W_out, ones(Int, 2, 10))
@test isequal(size(output), size(ones(Int, 2, 10)))
