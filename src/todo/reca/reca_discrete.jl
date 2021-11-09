"""
    RECAdirect_predict_discrete(reca::AbstractReca, W_out::AbstractArray{Float64}, test_data::AbstractArray{Int})

Given the input data return the corresponding predicted output, as described in [1].

[1] Yilmaz, Ozgur. “Reservoir computing using cellular automata.” arXiv preprint arXiv:1410.0162 (2014).
"""
function RECAdirect_predict_discrete(reca::AbstractReca, 
    W_out::AbstractArray{Float64}, 
    test_data::AbstractArray{Int})
    
    predict_len = size(test_data, 2)
    output = Array{Int}(undef, size(W_out, 1), predict_len)
    init_ca = zeros(Int, reca.expansion_size*reca.permutations)
    
    for i=1:predict_len
        init_ca = encoding(test_data[:,i], init_ca, reca.maps)
        ca = ECA(reca.rule, init_ca, reca.generations+1)
        ca_states = ca.cells[2:end ,:]
        x = copy(reshape(transpose(ca_states), reca.generations*reca.expansion_size*reca.permutations))
        out = W_out*x
        init_ca = ca.cells[end, :]
        output[:,i] = convert(AbstractArray{Int}, out .> 0.5)
    end
    return output
end
 
 
