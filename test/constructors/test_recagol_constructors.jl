using ReservoirComputing

train_data = ones(Int, 2, 10)
res_size = 10
generations = 5
permutations = 2

reca = RECA_TwoDim(train_data, res_size, generations, permutations)

@test isequal(reca.train_data, train_data)
@test isequal(reca.res_size, res_size)
@test isequal(reca.generations, generations)
@test isequal(reca.permutations, permutations)
