using ReservoirComputing

const approx_res_size = 12
const alpha = 1.0
const beta = 1*10^(-5)
const extended_states = false
const nla_type = NLADefault()
const activation = tanh

const input_weight = 0.1
const cyrcle_weight = 0.99
const jump_weight = 0.1
const jumps = 12

const memory_size = 16

input = ones(Float64, 10, 2)
output = ones(Float64, 10, 2)

W = CRJ(approx_res_size, cyrcle_weight, jump_weight, jumps)
W_in = irrational_sign_input(approx_res_size, size(input, 2), input_weight)

rmm = RMM(W, input, output, W_in, memory_size, beta,
        extended_states = extended_states,
        nla_type = nla_type,
        activation = activation,
        alpha = alpha)

@test isequal(approx_res_size, rmm.res_size)
@test isequal(rmm.in_size, size(input, 2))
@test isequal(rmm.out_size, size(output, 2))
@test isequal(input, rmm.train_data)
@test isequal(alpha, rmm.alpha)
@test isequal(nla_type, rmm.nla_type)
@test isequal(activation, rmm.activation)
@test size(rmm.W) == (rmm.res_size, rmm.res_size)
@test size(rmm.W_in) == (rmm.res_size, rmm.in_size)
@test isequal(rmm.extended_states, extended_states)
@test size(rmm.W_out) == (rmm.out_size, rmm.res_size+rmm.in_size)
#test direct predict
rmmout = RMMdirect_predict(rmm, input)
@test size(rmmout) == size(input)


