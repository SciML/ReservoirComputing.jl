using ReservoirComputing
using ParameterizedFunctions
using DifferentialEquations
using Plots

#lorenz system parameters
u0 = [1.0,0.0,0.0]                       
tspan = (0.0,200.0)                      
p = [10.0,28.0,8/3]
#model parameters
shift = 1
approx_res_size = 300
N = 3
radius = 1.2
degree = 6
sigma = 0.1
in_size = N
out_size = N
train_len = 5000
predict_len = 1250
beta = 0.0
alpha = 1.0
nonlin_alg = "T2"

#define lorenz system 
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end
    
prob = ODEProblem(lorenz, u0, tspan, p)  
sol = solve(prob, AB4(), dt=0.02)   
v = sol.u
data = Matrix(hcat(v...))
train = data[:, shift:shift+train_len-1]
test = data[:, train_len:train_len+predict_len-1]



#create echo state network  
esn = ESN(approx_res_size,
    in_size,
    out_size,
    train,
    degree,
    sigma,
    alpha,
    beta,
    radius,
    nonlin_alg)

W_out = ESNtrain(esn)
output = ESNsingle_predict(esn, predict_len, test[3,:], test, W_out)

#plots and images
comp = plot(transpose(output),layout=(3,1), label="predicted")
comp = plot!(transpose(test),layout=(3,1), label="actual")
savefig(comp, "single_comp")

sf = plot(transpose(output)[:,1], transpose(output)[:,2], transpose(output)[:,3], label="predicted")
sf2 = plot(transpose(test)[:,1], 
           transpose(test)[:,2], 
           transpose(test)[:,3], 
           label="actual")
           
final = plot(sf, sf2, layout=(1,2), label = ["predicted" "actual"])
savefig(final, "single_attractor_com") 
