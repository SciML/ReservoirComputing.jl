using Plots, OrdinaryDiffEq, ReservoirComputing, Random
using Plots.PlotMeasures


Random.seed!(4242)
#define lorenz system
function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end
#solve and take data
prob = ODEProblem(lorenz!, [1.0,0.0,0.0], (0.0,200.0))
data = solve(prob, ABM54(), dt=0.02)

shift = 300
train_len = 5000
predict_len = 1250
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]
test_data = data[:,shift+train_len+1:shift+train_len+predict_len]

esn = ESN(300, input_data; 
    variation = Default(),
    reservoir_init = RandSparseReservoir(radius=1.2, sparsity=6/300),
    input_init = WeightedLayer(),
    reservoir_driver = RNN(),
    nla_type = NLADefault(),
    states_type = StandardStates())

training_method = StandardRidge(0.0) 
output_layer = train(esn, target_data, training_method)

output = esn(Generative(predict_len), output_layer)

ts = 0.0:0.02:200.0
lorenz_maxlyap = 0.9056
predict_ts = ts[shift+train_len+1:shift+train_len+predict_len]
lyap_time = (predict_ts .- predict_ts[1])*(1/lorenz_maxlyap)

p1 = plot(lyap_time, [test_data[1,:] output[1,:]], label = ["actual" "predicted"], 
    ylabel = "x(t)", linewidth=2.5, xticks=false, yticks = -15:15:15);
p2 = plot(lyap_time, [test_data[2,:] output[2,:]], label = ["actual" "predicted"], 
    ylabel = "y(t)", linewidth=2.5, xticks=false, yticks = -20:20:20);
p3 = plot(lyap_time, [test_data[3,:] output[3,:]], label = ["actual" "predicted"], 
    ylabel = "z(t)", linewidth=2.5, xlabel = "max(λ)*t", yticks = 10:15:40);


plot(p1, p2, p3, size=(1080, 720), plot_title = "Lorenz System Coordinates", 
    layout=(3,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20, left_margin=4mm)
savefig("lorenz_basic.png")
