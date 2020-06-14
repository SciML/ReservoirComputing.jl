using LIBSVM
using ReservoirComputing
using Plots
using ParameterizedFunctions
using DifferentialEquations
using DynamicalSystems
using Statistics

function nrmse(y, yt, sigma)
    rmse = 0.0
    for i=1:size(y, 1)
        rmse += (y[i]-yt[i])^2.0
    end
    rmse = sqrt(rmse/(size(y, 1)*sigma))
    return rmse
end

function data_prep(data, shift, train_len, test_len, h)
    
    new_d = Matrix(embed(Dataset(data), 4, 6))'
    y_target = data[h:end]
    train_in, train_out = new_d[:, shift:shift+train_len-1], y_target[shift:shift+train_len-1]
    test_in, test_out = new_d[:, shift+train_len:shift+train_len+test_len-1], y_target[shift+train_len:shift+train_len+test_len-1]
    
    return train_in, train_out, test_in, test_out
end

function f_mackey_glass(u, h, p, t)
  z = h(p, t - 17)

  0.2 * z / (1 + z^10) - 0.1 * u
end

function h_mackey_glass(p, t)
  t ≤ 0 || error("history function is only implemented for t ≤ 0")

  0.5
end

prob = DDEProblem(f_mackey_glass, h_mackey_glass, (0.0, 700.0); constant_lags = [17])
sol = solve(prob, RK4(), adaptive=false,dt=0.1)
v = sol.u
data = Matrix(hcat(v...)')

#data parameters
const shift = 200
const train_len = 1000
const test_len = 500
const h = 84

#model parameters
const approx_res_size = 700
const sparsity = 0.02
const activation = tanh
const radius = 0.98
const sigma = 0.25

const alpha = 1.0
const nla_type = NLADefault()
const extended_states = true

#creating data
train_in, train_out, test_in, test_out = data_prep(data, shift, train_len, test_len, h)


#add noise on training and testing step
println("Training and testing on noisy data...")
test_in += 0.045*randn(size(test_in, 1), size(test_in, 2))
train_in += 0.045*randn(size(train_in, 1), size(train_in, 2))

#create echo state network  
W = init_reservoir_givensp(approx_res_size, radius, sparsity)
W_in = init_dense_input_layer(approx_res_size, size(train_in, 1), sigma)
esn = ESN(W, train_in, W_in, activation, alpha, nla_type, extended_states)
#train and predict using svesm
m = SVESMtrain(EpsilonSVR(kernel = Kernel.Linear), esn, train_out)

output = SVESM_direct_predict(esn, test_in, m)
#computing nrmse normalized on the variance of the original time series
sig = var(data)
println("SVESM nmrse for noisy training and testing: ", nrmse(test_out, output, sig))
svesm_nn = plot(output, label="SVESM")
svesm_nn = plot!(test_out, label="actual")
savefig(svesm_nn, "svesm_noisytraintest_comparison")

#svm training
m1 = fit!(EpsilonSVR(kernel = Kernel.Polynomial), train_in', train_out)
t1 = predict(m1, test_in')

m2 = fit!(EpsilonSVR(kernel = Kernel.RadialBasis), train_in', train_out)
t2 = predict(m2, test_in')

println("SVM Poly kernel nmrse for noisy training and testing: ", nrmse(test_out, t1, sig))
println("SVM RadialBasis kernel nmrse for noisy training and testing: ", nrmse(test_out, t2, sig))
svm_nn = plot(t1, label="SVM Polynomial kernel")
svm_nn = plot!(t2, label="SVM RadialBasis kernel")
svm_nn = plot!(test_out, label="actual")
savefig(svm_nn, "svm_noisytraintest_comparison")

