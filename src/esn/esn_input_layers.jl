

"""
    WeightedInput(scaling)
    WeightedInput(;scaling=0.1)

Returns a weighted layer initializer object, that when given as an input to the ```ESN``` call to  
```input_init``` will produce a weighted input matrix with a with random non-zero elements drawn 
from \$ [-\text{scaling}, \text{scaling}] \$, as described in [1]. The ```scaling``` factor can be 
given as arg or kwarg.

[1] Lu, Zhixin, et al. "Reservoir observers: Model-free inference of unmeasured variables in chaotic 
systems." Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
"""
struct WeightedLayer{T} <: AbstractLayer
    scaling::T
end

function WeightedLayer(; scaling=0.1)
    WeightedLayer(scaling)
end

function create_layer(approx_res_size, in_size, input_layer::WeightedLayer)

    res_size = Int(floor(approx_res_size/in_size)*in_size)
    input_matrix = zeros(res_size, in_size)
    q = floor(Int, res_size/in_size) #need to fix the reservoir input size. Check the constructor #should be fine now
    for i=1:in_size
        input_matrix[(i-1)*q+1 : (i)*q, i] = rand(Uniform(-input_layer.scaling, input_layer.scaling), 1, q)
    end
    input_matrix

end

"""
    DenseLayer(scaling)
    DenseLayer(;scaling=0.1)

Returns a fully connected layer initializer object, that when given as an input to the ```ESN``` 
call to  ```input_init``` will produce a weighted input matrix with a with random non-zero elements
 drawn from \$ [-\text{scaling}, \text{scaling}] \$. The ```scaling``` factor can be given as arg or
 kwarg.
"""
struct DenseLayer{T} <: AbstractLayer
    scaling::T
end

function DenseLayer(; scaling=0.1)
    DenseLayer(scaling)
end

"""
    create_layer(res_size, in_size, input_layer::AbstractLayer)

Returns a res_size times in_size input layer, constructed accordingly to the ```input_layer``` 
constructor
"""
function create_layer(res_size, in_size, input_layer::DenseLayer)

    rand(Uniform(-input_layer.scaling, input_layer.scaling), res_size, in_size)
end

"""
    SparseLayer(scaling, sparsity)
    SparseLayer(scaling; sparsity=0.1)
    SparseLayer(;scaling=0.1, sparsity=0.1)

Returns a sparsely connected layer initializer object, that when given as an input to the ```ESN```
 call to  ```input_init``` will produce a random sparse input matrix with random non-zero elements 
 drawn from \$ [-\text{scaling}, \text{scaling}] \$ and given sparsity. The ```scaling``` and 
 ```sparsity``` factors can be given as an arg or kwarg.
"""
struct SparseLayer{T} <: AbstractLayer
    scaling::T
    sparsity::T
end

function SparseLayer(; scaling=0.1, sparsity=0.1)
    SparseLayer(scaling, sparsity)
end

function SparseLayer(scaling_arg; scaling=scaling_arg, sparsity=0.1)
    SparseLayer(scaling, sparsity)
end

function create_layer(res_size, in_size, input_layer::SparseLayer)

    input_matrix = Matrix(sprand(res_size, in_size, input_layer.sparsity))
    input_matrix = 2.0 .*(input_matrix.-0.5)
    replace!(input_matrix, -1.0=>0.0)
    input_matrix = input_layer.scaling .*input_matrix
    input_matrix
end

#from "minimum complexity echo state network" Rodan
#and "simple deterministically constructed cycle reservoirs with regular jumps" by Rodan and Tino

"""
    BernoulliSample(p)
    BernoulliSample(;p=0.5)

Returns a Bernoulli sign constructor for the ```MinimumLayer``` call. The ```p``` factor determines the 
probability of the result as in the Distributions call. The value can be passed as an arg or kwarg.
"""
struct BernoulliSample{T}
    p::T
end

function BernoulliSample(;p=0.5)
    BernoulliSample(p)
end

"""
    IrrationalSample(irrational, start)
    IrrationalSample(;irrational=pi, start=1)

Returnsan irational sign contructor for the '''MinimumLayer''' call. The values can be passed as args or 
kwargs.
"""
struct IrrationalSample{K}
    irrational::Irrational
    start::K
end

function IrrationalSample(;irrational=pi, start=1)
    IrrationalSample(irrational, start)
end

"""
    MinimumLayer(weight, sampling)
    MinimumLayer(;weight=0.1, sampling=BernoulliSample())

Returns a fully connected layer initializer object, where all the weights are the same, decided by 
the ```weight``` factor and the sign of each entry is decided by the ```sampling``` struct. 
Construction detailed in [1] and [2].

[1] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on 
neural networks 22.1 (2010): 131-144.
[2] Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs with regular 
jumps." Neural computation 24.7 (2012): 1822-1852.
"""
struct MinimumLayer{T,K} <: AbstractLayer
    weight::T
    sampling::K
end

function MinimumLayer(weight; sampling=BernoulliSample(0.5))
    MinimumLayer(weight, sampling)
end

function MinimumLayer(; weight=0.1, sampling=BernoulliSample(0.5))
    MinimumLayer(weight, sampling)
end

function create_layer(res_size, in_size, input_layer::MinimumLayer)

    input_matrix = create_minimum_input(res_size, in_size, input_layer.weight, input_layer.sampling)
end

function create_minimum_input(res_size, in_size, weight, sampling::BernoulliSample)
    input_matrix = zeros(res_size, in_size)
    for i=1:res_size
        for j=1:in_size
            rand(Bernoulli(sampling.p)) ? input_matrix[i, j] = weight : input_matrix[i, j] = -weight                
        end
    end
    input_matrix
end

function create_minimum_input(res_size, in_size, weight, sampling::IrrationalSample)

    setprecision(BigFloat, Int(ceil(log2(10)*(res_size*in_size+sampling.start+1))))
    ir_string = string(BigFloat(sampling.irrational)) |> collect
    deleteat!(ir_string, findall(x->x=='.', ir_string))
    ir_array = zeros(length(ir_string))
    input_matrix = zeros(res_size, in_size)

    for i =1:length(ir_string)
        ir_array[i] = parse(Int, ir_string[i])
    end

    counter = sampling.start

    for i=1:res_size
        for j=1:in_size
            ir_array[counter] < 5 ? input_matrix[i, j] = -weight : input_matrix[i, j] = weight
            counter += 1
        end
    end
    input_matrix
end

"""
physics_informed_input(res_size::Int, in_size::Int, sigma::Float64, γ::Float64)

Return a weighted input layer matrix, with random non-zero elements drawn from \$ [-\\text{sigma}, \\text{sigma}] \$, where some γ
of reservoir nodes are connected exclusively to the raw inputs, and the rest to the outputs of the prior knowledge model , as described in [1].

[1] Jaideep Pathak et al. "Hybrid Forecasting of Chaotic Processes: Using Machine Learning in Conjunction with a Knowledge-Based Model" (2018)
"""

struct InformedLayer{T,K,M} <: AbstractLayer
    scaling::T
    gamma::K
    model_in_size::M
end

function InformedLayer(model_in_size; scaling=0.1, gamma=0.5)
    InformedLayer(scaling, gamma, model_in_size)
end

function create_layer(res_size, in_size, input_layer::InformedLayer)

    state_size = in_size - input_layer.model_in_size
    if state_size <= 0
        throw(DimensionMismatch("in_size must be greater than model_in_size"))
    end
    input_matrix = zeros(res_size, in_size)
    #Vector used to find res nodes not yet connected
    zero_connections = zeros(in_size)
    #Num of res nodes allotted for raw states
    num_for_state = floor(Int, res_size*input_layer.gamma)
    #Num of res nodes allotted for prior model input
    num_for_model = floor(Int, (res_size*(1-input_layer.gamma)))
    for i in 1:num_for_state
        #find res nodes with no connections
        idxs = findall(Bool[zero_connections == input_matrix[i,:] for i=1:size(input_matrix,1)])
        random_row_idx = idxs[rand(1:end)]
        random_clm_idx = range(1, state_size, step = 1)[rand(1:end)]
        input_matrix[random_row_idx,random_clm_idx] = rand(Uniform(-input_layer.scaling, input_layer.scaling))
    end

    for i in 1:num_for_model
        idxs = findall(Bool[zero_connections == input_matrix[i,:] for i=1:size(input_matrix,1)])
        random_row_idx = idxs[rand(1:end)]
        random_clm_idx = range(state_size+1, in_size, step = 1)[rand(1:end)]
        input_matrix[random_row_idx,random_clm_idx] = rand(Uniform(-input_layer.scaling, input_layer.scaling))
    end
    input_matrix
end
