function closed_form_sum(n, p)
    #M = 1 + n + ... + n^p
    #M = \sum_{m=0}^{p}n^m = \frac{n^{p+1}-1}{n-1}
    return Int((n^(p+1)-1)/(n-1))
end

function poly_vector(input_vector::AbstractArray{Float64}, 
        polynomial_order::Int)
    
    size_poly_vector = closed_form_sum(size(input_vector, 1), polynomial_order)    
    poly_vector = ones(Float64, size_poly_vector)
    
    counter = 1
    for i=1:polynomial_order
        var = size(input_vector, 1)^i
        for j=1:var
            counter+=1
            poly_vector[counter] = poly_vector[Int(i-1+ceil(j/size(input_vector, 1)))]*input_vector[mod1(j, size(input_vector, 1))]
        end
    end
    return poly_vector
end
