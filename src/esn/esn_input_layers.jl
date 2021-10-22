

"""
    init_input_layer(res_size::Int, in_size::Int, sigma::Float64)

Return a weighted input layer matrix, with random non-zero elements drawn from \$ [-\\text{sigma}, \\text{sigma}] \$, as described in [1].

[1] Lu, Zhixin, et al. "Reservoir observers: Model-free inference of unmeasured variables in chaotic systems." Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
"""

struct WeightedInput{T} <: AbstractInputLayer
    scaling::T
end

function WeightedInput(; scaling=0.1)
    WeightedInput(scaling)
end

function create_layer(approx_res_size, in_size, input_layer::WeightedInput)

    res_size = Int(floor(approx_res_size/in_size)*in_size)
    input_matrix = zeros(res_size, in_size)
    q = floor(Int, res_size/in_size) #need to fix the reservoir input size. Check the constructor
    for i=1:in_size
        input_matrix[(i-1)*q+1 : (i)*q, i] = (2*input_layer.scaling).*(rand(1, q).-0.5)
    end
    input_matrix

end

"""
    init_dense_input_layer(res_size::Int, in_size::Int, sigma::Float64)

Return a fully connected input layer matrix, with random non-zero elements drawn from \$ [-sigma, sigma] \$.
"""

struct DenseInput{T} <: AbstractInputLayer
    scaling::T
end

function DenseInput(; scaling=0.1)
    DenseInput(scaling)
end

function create_layer(res_size, in_size, input_layer::DenseInput)

    input_matrix = rand(res_size, in_size)
    input_matrix = 2.0 .*(input_matrix.-0.5)
    input_matrix = input_layer.scaling .*input_matrix
    input_matrix
end

"""
    init_sparse_input_layer(res_size::Int, in_size::Int, sigma::Float64, sparsity::Float64)

Return a sparsely connected input layer matrix, with random non-zero elements drawn from \$ [-sigma, sigma] \$ and given sparsity.
"""

struct SparseInput{T} <: AbstractInputLayer
    scaling::T
    sparsity::T
end

function SparseInput(; scaling=0.1, sparsity=0.1)
    SparseInput(scaling, sparsity)
end

function SparseInput(scaling_arg; scaling=scaling_arg, sparsity=0.1)
    SparseInput(scaling, sparsity)
end

function create_layer(res_size, in_size, input_layer::SparseInput)

    input_matrix = Matrix(sprand(res_size, in_size, input_layer.sparsity))
    input_matrix = 2.0 .*(input_matrix.-0.5)
    replace!(input_matrix, -1.0=>0.0)
    input_matrix = input_layer.scaling .*input_matrix
    input_matrix
end

#from "minimum complexity echo state network" Rodan
#and "simple deterministically constructed cycle reservoirs with regular jumps" by Rodan and Tino
"""
    irrational_sign_input(res_size::Int, in_size::Int , weight::Float64 [, start::Int, irrational::Irrational])

Return a fully connected input layer matrix with the same weights and sign decided by the values of an irrational number, as described in [1] and [2].

[1] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
[2] Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""
struct BernoulliSample{T}
    p::T
end

struct IrrationalSample{K}
    irrational::Irrational
    start::K
end

function IrrationalSample(;irrational=pi, start=1)
    IrrationalSample(irrational, start)
end

struct MinimumInput{T,K} <: AbstractInputLayer
    weight::T
    sampling::K
end

function MinimumInput(weight; sampling=BernoulliSample(0.5))
    MinimumInput(weight, sampling)
end

function MinimumInput(; weight=0.1, sampling=BernoulliSample(0.5))
    MinimumInput(weight, sampling)
end

function create_layer(res_size, in_size, input_layer::MinimumInput)

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

struct InformedInput{T,K,M} <: AbstractInputLayer
    scaling::T
    gamma::K
    model_in_size::M
end

function InformedInput(model_in_size; scaling=0.1, gamma=0.5)
    InformedInput(scaling, gamma, model_in_size)
end

function create_layer(res_size, in_size, input_layer::InformedInput)

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
