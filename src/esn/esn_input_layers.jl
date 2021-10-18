

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

function create_input_layer(approx_res_size, in_size, input_layer::WeightedInput)

    res_size = Int(floor(approx_res_size/in_size)*in_size)
    input_matrix = zeros(Float64, res_size, in_size)
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

function create_input_layer(res_size, in_size, input_layer::DenseInput)

    input_matrix = rand(Float64, res_size, in_size)
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

function create_input_layer(res_size, in_size, input_layer::SparseInput)

    input_matrix = Matrix(sprand(Float64, res_size, in_size, input_layer.sparsity))
    input_matrix = 2.0 .*(input_matrix.-0.5)
    replace!(input_matrix, -1.0=>0.0)
    input_matrix = input_layer.scaling .*input_matrix
    input_matrix
end

#from "minimum complexity echo state network" Rodan
"""
    min_complex_input(res_size::Int, in_size::Int, weight::Float64)

Return a fully connected input layer matrix with the same weights and sign drawn from a Bernoulli distribution, as described in [1].

[1] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
"""

struct MinimumInput{T} <: AbstractInputLayer
    weight::T
end

function MinimumInput(; weight=0.1)
    MinimumInput(weight)
end

function create_input_layer(res_size, in_size, input_layer::MinimumInput)

    input_matrix = Array{Float64}(undef, res_size, in_size)
    for i=1:res_size
        for j=1:in_size
            if rand(Bernoulli()) == true
                input_matrix[i, j] = input_layer.weight
            else
                input_matrix[i, j] = -input_layer.weight
            end
        end
    end
    input_matrix
end

#from "minimum complexity echo state network" Rodan
#and "simple deterministically constructed cycle reservoirs with regular jumps" by Rodan and Tino
#=
"""
    irrational_sign_input(res_size::Int, in_size::Int , weight::Float64 [, start::Int, irrational::Irrational])

Return a fully connected input layer matrix with the same weights and sign decided by the values of an irrational number, as described in [1] and [2].

[1] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.
[2] Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""
function irrational_sign_input(res_size, in_size; weight=0.1; start = 1, irrational::Irrational = pi) #to fix

    setprecision(BigFloat, Int(ceil(log2(10)*(res_size*in_size+start+1))))
    ir_string = string(BigFloat(irrational)) |> collect
    deleteat!(ir_string, findall(x->x=='.', ir_string))
    ir_array = Array{Int}(undef, length(ir_string))
    W_in = Array{Float64}(undef, res_size, in_size)

    for i =1:length(ir_string)
        ir_array[i] = parse(Int, ir_string[i])
    end

    counter = start

    for i=1:res_size
        for j=1:in_size
            if ir_array[counter] < 5
                W_in[i, j] = -weight
            else
                W_in[i, j] = weight
            end
            counter += 1
        end
    end
    return W_in
end


"""
physics_informed_input(res_size::Int, in_size::Int, sigma::Float64, γ::Float64)

Return a weighted input layer matrix, with random non-zero elements drawn from \$ [-\\text{sigma}, \\text{sigma}] \$, where some γ
of reservoir nodes are connected exclusively to the raw inputs, and the rest to the outputs of the prior knowledge model , as described in [1].

[1] Jaideep Pathak et al. "Hybrid Forecasting of Chaotic Processes: Using Machine Learning in Conjunction with a Knowledge-Based Model" (2018)
"""
function PhyInfInput(res_size, in_size; sigma=0.1, γ=0.1, model_in_size=in_size) #to check

    state_size = in_size - model_in_size
    if state_size <= 0
        throw(DimensionMismatch("in_size must be greater than model_in_size"))
    end
    W_in = zeros(Float64, res_size, in_size)
    #Vector used to find res nodes not yet connected
    zero_connections = zeros(in_size)
    #Num of res nodes allotted for raw states
    num_for_state = floor(Int, res_size*γ)
    #Num of res nodes allotted for prior model input
    num_for_model = floor(Int, (res_size*(1-γ)))
    for i in 1:num_for_state
        #find res nodes with no connections
        idxs = findall(Bool[zero_connections == W_in[i,:] for i=1:size(W_in,1)])
        random_row_idx = idxs[rand(1:end)]
        random_clm_idx = range(1, state_size, step = 1)[rand(1:end)]
        W_in[random_row_idx,random_clm_idx] = rand(Uniform(-sigma, sigma))
    end

    for i in 1:num_for_model
        idxs = findall(Bool[zero_connections == W_in[i,:] for i=1:size(W_in,1)])
        random_row_idx = idxs[rand(1:end)]
        random_clm_idx = range(state_size+1, in_size, step = 1)[rand(1:end)]
        W_in[random_row_idx,random_clm_idx] = rand(Uniform(-sigma, sigma))
    end
    return W_in
end
=#