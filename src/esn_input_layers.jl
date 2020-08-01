function init_input_layer(res_size::Int,
        in_size::Int,
        sigma::Float64)

    W_in = zeros(Float64, res_size, in_size)
    q = floor(Int, res_size/in_size) #need to fix the reservoir input size. Check the constructor
    for i=1:in_size
        W_in[(i-1)*q+1 : (i)*q, i] = (2*sigma).*(rand(Float64, 1, q).-0.5)
    end
    return W_in

end 

function init_dense_input_layer(res_size::Int,
        in_size::Int,
        sigma::Float64)

    W_in = rand(Float64, res_size, in_size)
    W_in = 2.0 .*(W_in.-0.5)
    W_in = sigma .*W_in
    return W_in
end

function init_sparse_input_layer(res_size::Int,
        in_size::Int,
        sigma::Float64,
        sparsity::Float64)

    W_in = Matrix(sprand(Float64, res_size, in_size, sparsity))
    W_in = 2.0 .*(W_in.-0.5)
    replace!(W_in, -1.0=>0.0)
    W_in = sigma .*W_in
    return W_in
end

#from "minimum complexity echo state network" Rodan 
function min_complex_input(res_size::Int,
        in_size::Int,
        weight::Float64)
    
    W_in = Array{Float64}(undef, res_size, in_size)
    for i=1:res_size
        for j=1:in_size
            if rand(Bernoulli()) == true
                W_in[i, j] = weight
            else
                W_in[i, j] = -weight
            end
        end
    end
    return W_in
end 

#from "minimum complexity echo state network" Rodan 
#and "simple deterministically constructed cycle reservoirs with regular jumps" by Rodan and Tino
function irrational_sign_input(res_size::Int,
        in_size::Int,
        weight::Float64;
        irrational::Irrational = pi)
    
    setprecision(BigFloat, Int(ceil(log2(10)*(res_size*in_size+1))))
    ir_string = string(BigFloat(irrational)) |> collect
    deleteat!(ir_string, findall(x->x=='.', ir_string))
    ir_array = Array{Int}(undef, length(ir_string))
    W_in = Array{Float64}(undef, res_size, in_size)

    for i =1:length(ir_string)
        ir_array[i] = parse(Int, ir_string[i])
    end    
    
    counter = 0
    
    for i=1:res_size
        for j=1:in_size
            counter += 1
            println
            if ir_array[counter] < 5
                W_in[i, j] = -weight
            else
                W_in[i, j] = weight
            end
        end
    end
    return W_in
end
