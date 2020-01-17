using EchoStateNetwork
using Plots

#lorenz system parameters
u0 = [1.0,0.0,0.0]                       
tspan = (0.0,125.0)                      
p = [10.0,28.0,8/3]
#model parameters
shift = 1
approx_res_size = 300
N = 3
res_size = Int(floor(approx_res_size/N)*N)
radius = 1.2
degree = 6
sigma = 0.1
in_size = N
out_size = N
train_len = 5000
predict_len = 1250
beta = 0.0
alpha = 1.0
filepath = "/home/nuzdotbot/Desktop/uni/esn/julia/lorenz.txt"
nonlin_alg = "T2"

data, datan = create_data(u0, tspan, p, shift, train_len)
W = init_reservoir(res_size, radius, degree)
W_in = init_input_layer(res_size, in_size, sigma)
states = states_matrix(W, W_in, data, res_size, train_len, alpha)
W_out = esn_train(beta, res_size, states, data, nonlin_alg)
output = esn_predict(in_size, predict_len, W_in, W, W_out, states, alpha, nonlin_alg)

comp = plot(transpose(output),layout=(3,1), label="predicted")
comp = plot!(transpose(datan[:, train_len:train_len+predict_len-1]),layout=(3,1), label="actual")
savefig(comp, "comp")
