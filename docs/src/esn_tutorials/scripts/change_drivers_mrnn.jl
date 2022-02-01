using ReservoirComputing, StatsBase, Random

Random.seed!(42)

u(t) = sin(t)+sin(0.51*t)+sin(0.22*t)+sin(0.1002*t)+sin(0.05343*t)

train_len = 3000
predict_len = 2000
shift = 1

data = u.(collect(0.0:0.01:500))
training_input = reduce(hcat, data[shift:shift+train_len-1])
training_target = reduce(hcat, data[shift+1:shift+train_len])
testing_input = reduce(hcat, data[shift+train_len:shift+train_len+predict_len-1])
testing_target = reduce(hcat, data[shift+train_len+1:shift+train_len+predict_len])

f2(x) = (1-exp(-x))/(2*(1+exp(-x)))
f3(x) = (2/pi)*atan((pi/2)*x)
f4(x) = x/sqrt(1+x*x)

base_case = RNN(tanh, 0.85)

case3 = MRNN(activation_function=[tanh, f2], 
    leaky_coefficient=0.85, 
    scaling_factor=[0.5, 0.3])

case4 = MRNN(activation_function=[tanh, f3], 
    leaky_coefficient=0.9, 
    scaling_factor=[0.45, 0.35])

case5 = MRNN(activation_function=[tanh, f4], 
    leaky_coefficient=0.9, 
    scaling_factor=[0.43, 0.13])

test_cases = [base_case, case3, case4, case5]

for case in test_cases
    esn = ESN(100, training_input,
        input_init = WeightedLayer(scaling=0.3),
        reservoir_init = RandSparseReservoir(radius=0.4),
        reservoir_driver = case,
        states_type = ExtendedStates())
    wout = train(esn, training_target, StandardRidge(10e-6))
    output = esn(Predictive(testing_input), wout)
    println(rmsd(testing_target, output, normalize=true))
end
