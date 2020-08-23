abstract type AbstractReservoirMemoryMachine <: AbstractEchoStateNetwork end

mutable struct RMM{T, S<:AbstractArray{T}, I, B, F, N} <: AbstractReservoirMemoryMachine
    res_size::I
    in_size::I
    out_size::I
    train_data::S
    alpha::T
    nla_type::N
    activation::F
    W::S
    W_in::S
    states::S
    extended_states::B
    W_out::S
    memory_size::I
    write_matrix::S
    read_matrix::S
end

"""
    RMM(W::AbstractArray{T}, in_data::AbstractArray{T}, out_data::AbstractArray{T}, W_in::AbstractArray{T}, 
    memory_size::Int, beta::Float64 [, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])
    
Create a Reservoir Memory Machine struct, as described in [1]

[1] Paaßen, Benjamin, and Alexander Schulz. "Reservoir memory machines." arXiv preprint arXiv:2003.04793 (2020).
"""
function RMM(W::AbstractArray{T},
        in_data::AbstractArray{T},
        out_data::AbstractArray{T},
        W_in::AbstractArray{T},
        memory_size::Int,
        beta::Float64;
        activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(in_data, 2)
    out_size = size(out_data, 2)
    res_size = size(W, 1)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch("size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch("size(W_in, 2) must be equal to in_size"))
    end

    states = states_matrix_inplace(W, W_in, collect(in_data'), alpha, activation, extended_states)
    write_matrix, read_matrix = identity_init(in_data, out_data, memory_size, states, beta)#forward_init(states, beta) #identity_init(input, output, memory_size, states, beta)
    memory_states = apply_write(in_data, states, write_matrix, memory_size)
    reads = apply_read(in_data, states, write_matrix, read_matrix, memory_size)
    wout = RMMtrain(in_data, out_data, states, write_matrix, read_matrix, memory_states, memory_size, beta)
    
    return RMM{T, typeof(in_data), 
        typeof(res_size), 
        typeof(extended_states), 
        typeof(activation), 
        typeof(nla_type)}(res_size, in_size, out_size, in_data,
    alpha, nla_type, activation, W, W_in, states, extended_states, wout, memory_size, write_matrix, read_matrix)
end

function RMMtrain(input, output, states, write_matrix, read_matrix, memory_states, memory_size, beta)
    
    states_pinv = (states*inv(states'*states+beta*Matrix(1.0I, size(states, 2), size(states, 2))))
    last_loss = Inf 
    last_vs = nothing
    
    for epoch=1:30
        reads = apply_read(input, states, write_matrix, read_matrix, memory_size)
        wout = (output'*reads)*inv(add_reg(reads'*reads, beta))
        loss = sqrt(mean(sum((output - reads*wout').^2, dims=2)))
        
        if loss < last_loss - 1*10^(-3)
            last_loss = loss
            last_vs = (write_matrix, read_matrix, wout)
        else
            write_matrix, read_matrix, wout = last_vs
            
            break
        end
    
        write_actions = align_write(input, output, wout, memory_size)
        write_matrix = write_actions'*states_pinv
        
        a, read_actions = align_read(memory_states, output, memory_size, wout)
        read_matrix = collect(reduce(hcat, read_actions))*states_pinv
    end
    
    read = apply_read(input, states, write_matrix, read_matrix, memory_size)
    hr = hcat(states, read)
    wout = (output'*hr)*inv(add_reg(hr'*hr, beta))
    return wout
    
end

"""
    RMMdirect_predict(rmm::AbstractReservoirMemoryMachine, input)

Given the input data return the corresponding predicted output, as described in [1].

[1] Paaßen, Benjamin, and Alexander Schulz. "Reservoir memory machines." arXiv preprint arXiv:2003.04793 (2020).
"""
function RMMdirect_predict(rmm::AbstractReservoirMemoryMachine, input)
    
    predict_len = size(input, 1)
    predict_states = zeros(Float64, predict_len, rmm.res_size)
    #compute first state
    predict_states[1, :] = rmm.activation.(rmm.W_in*input[1, :])

    if rmm.extended_states == false
        for i=2:predict_len
            predict_states[i, :] = leaky_fixed_rnn(rmm.activation, rmm.alpha, rmm.W, rmm.W_in, predict_states[i-1, :], input[i,:])
        end
    end
    
    reads = apply_read(input, predict_states, rmm.write_matrix, rmm.read_matrix, rmm.memory_size)
    predict_states = hcat(predict_states, reads)
    output = predict_states*rmm.W_out'
    return output
end

function states_matrix_inplace(W::AbstractArray{Float64},
        W_in::AbstractArray{Float64},
        train_data::AbstractArray{Float64},
        alpha::Float64,
        activation::Function,
        extended_states::Bool)

    train_len = size(train_data, 2)
    res_size = size(W, 1)
    in_size = size(train_data, 1)

    states = zeros(Float64, train_len, res_size)
    states[1, :] = activation.(W_in*train_data[:, 1])
    
    for i=2:train_len
        states[i, :] = leaky_fixed_rnn(activation, alpha, W, W_in, states[i-1, :], train_data[:, i])
    end

    if extended_states == true
        ext_states = vcat(states, hcat(zeros(Float64, in_size), train_data[:,1:end-1]))
        return ext_states
    else
        return states
    end
end

function apply_write(input, states, write_matrix, memory_size)
    
    in_size = size(input, 2)
    train_len = size(input, 1)
    write_actions = states*write_matrix'
    memory_states = zeros(Float64, memory_size, in_size, train_len)
    write_loc = 1
    
    for i=1:train_len
        if i > 1
            memory_states[:, :, i] = memory_states[:, :, i-1]
        end
        
        if write_actions[i] > 0
            memory_states[write_loc, :, i] = input[i,:]
            write_loc += 1
            if write_loc >= memory_size
                write_loc = 1
            end
        end
    end
    return memory_states
end

function apply_read(input, states, write_matrix, read_matrix, memory_size)
    
    train_len = size(input, 1)
    
    write_actions = states*write_matrix'
    read_actions =states*read_matrix'
    read_actions = mapslices(argmax, read_actions, dims=2)
    memory = zeros(memory_size, size(input, 2))
        
    write_loc = 1
    read_loc = 1
    reads = Array{Float64}(undef, train_len, size(input, 2))
    for i=1:train_len
        if write_actions[i] > 0
            memory[write_loc, :] = input[i,:]
            write_loc += 1
            if write_loc >= memory_size
                write_loc = 1
            end
        end
        
        if read_actions[i] == 2
            read_loc += 1
            if read_loc >= memory_size
                read_loc = 1
            end
        elseif read_actions[i] == 3
            read_loc = 1
        end
        
        reads[i, :] =  memory[read_loc, :]
    end
    
    return reads
end

function align_write(input, output, wout, memory_size)#  permit_duplicates always false. Input is Transpose than my implementation
    
    train_len = size(input, 1)
    delta = pairwise(Euclidean(), (input*transpose(wout)), output, dims = 1)
    delta_min = minimum(delta, dims = 1)
    
    writes = Array{Int}(undef, train_len)
    
    for i=1:train_len
        small_min =  findall(delta[:,i] .< minimum(delta[:,i])+1*10^(-3))[1]
        writes[i] = small_min#push!(writes, small_min)
    end
    
    write_loc = 1
    write_actions = zeros(Int, train_len)
    
    for i=1:train_len
        if i in writes
            write_loc += 1
            if write_loc >= memory_size
                write_loc = 1
            end
            write_actions[i] = 1
        else
            write_actions[i] = -1
        end
    end
    
    return write_actions
end

function align_read(memory_states, output, memory_size, wout)
    
    train_len = size(output, 1)
    delta = zeros(memory_size, train_len)
    for i=1:memory_size
        for j=1:train_len
            delta[i, j] = norm(wout*memory_states[i, :, j] - output[j, :])
        end
    end
    
    dyn_matrix = zeros(memory_size, train_len)
    for i=1:memory_size
        dyn_matrix[i, train_len-1] = delta[i, train_len-1]
    end
    
    for i in train_len-2: -1:1
        for j=1:memory_size-1
            dyn_matrix[j, i] = delta[j, i] +min(dyn_matrix[j, i+1], min(dyn_matrix[j+1, i+1], dyn_matrix[1, i+1]))
        end
        dyn_matrix[memory_size-1, i] = delta[memory_size-1, i] + min(dyn_matrix[memory_size-1, i+1], dyn_matrix[1, i+1])
    end
    
    actions = Array{Float64}[]
    if dyn_matrix[1, 1] < dyn_matrix[2, 1]+ 1*10^(-3)
        k = 1
        d = dyn_matrix[1, 1]
        push!(actions, [1, 0, 0])
    else
        k = 2
        d = dyn_matrix[2, 1]
        push!(actions, [0, 1, 0])
    end
    
    for i=1:train_len-1
        if delta[k, i] + dyn_matrix[k, i+1] < dyn_matrix[k, i] + 1*10^(-3)
            push!(actions, [1, 0, 0])
        elseif k < memory_size-1 && delta[k, i] + dyn_matrix[k+1, i+1] < dyn_matrix[k, i] + 1*10^(-3)
            push!(actions, [0, 1, 0])
            k+=1
        elseif delta[k, i] + dyn_matrix[1, i+1] < dyn_matrix[k, i] + 1*10^(-3)
            if k == memory_size-1
                push!(actions, [0, 1, 0])
            else
                push!(actions, [0, 0, 1])
            end
            k = 1
        else
            println("error")
        end
    end
    return min(dyn_matrix[1, 1], dyn_matrix[2, 1]), actions
end

function identity_init(input, output, memory_size, states, beta)
    
    states_pinv = (states*inv(states'*states+beta*Matrix(1.0I, size(states, 2), size(states, 2))))
    
    in_size = size(input, 2)
    train_len = size(input, 1)
    wout = Matrix{Float64}(I, in_size, in_size)
    #write_actions = []
    
    waj = align_write(input, output, wout, memory_size)
    #push!(write_actions, waj)
    write_matrix = collect(waj')*states_pinv
    
    memory_states = apply_write(input, states, write_matrix, memory_size)
    a, raj = align_read(memory_states, output, memory_size, wout)
    read_matrix = collect(reduce(hcat, raj))*states_pinv
    
    return write_matrix, read_matrix
end

function forward_init(states, beta)
    
    write_actions = ones(Float64, size(states, 1), 1)
    read_actions = zeros(Float64, size(states, 1), 3)
    read_actions[:,1] .= 1
    states_pinv = (states*inv(states'*states+beta*Matrix(1.0I, size(states, 2), size(states, 2))))
    write_matrix = write_actions'*states_pinv
    read_matrix  = read_actions'*states_pinv
    
    return write_matrix, read_matrix
end
