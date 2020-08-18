using ReservoirComputing

const approx_res_size = 10
const radius = 0.99
const sparsity = 0.1
const alpha = 1.0
const beta = 1*10^(-1)
const extended_states = false
const input_weight = 0.1
const gates_weight = 0.8
const nla_type = NLADefault()
const activation = tanh
const predict_len = 10


train = ones(Float64, 2, 10)

W = init_reservoir_givensp(approx_res_size, radius, sparsity)
W_in = irrational_sign_input(approx_res_size, size(train, 1), input_weight)

gruesn = GRUESN(W, train, W_in, 
        extended_states = extended_states, 
        gates_weight = gates_weight,
        nla_type = nla_type,
        activation = activation,
        alpha = alpha)
        
@test isequal(approx_res_size, gruesn.res_size)
@test isequal(gruesn.in_size, size(train, 1))
@test isequal(train, gruesn.train_data)
@test isequal(alpha, gruesn.alpha)
@test isequal(nla_type, gruesn.nla_type)
@test isequal(activation, gruesn.activation)
@test size(gruesn.W) == (gruesn.res_size, gruesn.res_size)
@test size(gruesn.W_in) == (gruesn.res_size, gruesn.in_size)
@test isequal(gruesn.extended_states, extended_states)

#test train
W_out = ESNtrain(gruesn, beta)
@test size(W_out) == (gruesn.in_size, gruesn.res_size)
#test predict
output = GRUESNpredict(gruesn, predict_len, W_out)
@test size(output) == (gruesn.in_size, predict_len)

