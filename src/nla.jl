
"""
    NLADefault()
Return the array untouched, default option.
"""
struct NLADefault <: NonLinearAlgorithm end

function nla(::NLADefault, x)
    return x
end

"""
    NLAT1()
Apply the \$ \\text{T}_1 \$ transformation algorithm, as defined in [1] and [2].

[1] Chattopadhyay, Ashesh, et al. "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).

[2] Pathak, Jaideep, et al. "Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach." Physical review letters 120.2 (2018): 024102.
"""
struct NLAT1 <: NonLinearAlgorithm end

function nla(::NLAT1, x_old)
    x_new = copy(x_old)
    for i =1:size(x_new, 1)
        if mod(i, 2)!=0
            x_new[i,:] = copy(x_old[i,:].*x_old[i,:])
        end
    end
    x_new
end

"""
    NLAT2()
Apply the \$ \\text{T}_2 \$ transformation algorithm, as defined in [1].

[1] Chattopadhyay, Ashesh, et al. "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).
"""
struct NLAT2 <: NonLinearAlgorithm end

function nla(::NLAT2, x_old)
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i,:] = copy(x_old[i-1,:].*x_old[i-2,:])
        end
    end
    x_new
end

"""
    NLAT3()
Apply the \$ \\text{T}_3 \$ transformation algorithm, as defined in [1].

[1] Chattopadhyay, Ashesh, et al. "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).
"""
struct NLAT3 <: NonLinearAlgorithm end

function nla(::NLAT3, x_old)
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i,:] = copy(x_old[i-1,:].*x_old[i+1,:])
        end
    end
    x_new
end
