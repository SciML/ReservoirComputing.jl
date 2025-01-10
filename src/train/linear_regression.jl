@doc raw"""

    StandardRidge([Type], [reg])

Returns a training method for `train` based on ridge regression.
The equations for ridge regression are as follows:

```math
\mathbf{w} = (\mathbf{X}^\top \mathbf{X} + 
\lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
```

# Arguments
 - `Type`: type of the regularization argument. Default is inferred internally,
   there's usually no need to tweak this
 - `reg`: regularization coefficient. Default is set to 0.0 (linear regression).

# Examples
```jldoctest
julia> ridge_reg = StandardRidge()
StandardRidge(0.0)

julia> ol = train(ridge_reg, rand(Float32, 10, 10), rand(Float32, 10, 10))
OutputLayer successfully trained with output size: 10

julia> ol.output_matrix #visualize output matrix
10×10 Matrix{Float32}:
  0.456574   -0.0407612   0.121963   …   0.859327   -0.127494    0.0572494
  0.133216   -0.0337922   0.0185378      0.24077     0.0297829   0.31512
  0.379672   -1.24541    -0.444314       1.02269    -0.0446086   0.482282
  1.18455    -0.517971   -0.133498       0.84473     0.31575     0.205857
 -0.119345    0.563294    0.747992       0.0102919   1.509      -0.328005
 -0.0716812   0.0976365   0.628654   …  -0.516041    2.4309     -0.113402
  0.0153872  -0.52334     0.0526867      0.729326    2.98958     1.32703
  0.154027    0.6013      1.05548       -0.0840203   0.991182   -0.328555
  1.11007    -0.0371736  -0.0529418      0.186796   -1.21815     0.204838
  0.282996   -0.263799    0.132079       0.875417    0.497951    0.273423

julia> ridge_reg = StandardRidge(0.001) #passing a value 
StandardRidge(0.001)

julia> ol = train(ridge_reg, rand(Float16, 10, 10), rand(Float16, 10, 10))
OutputLayer successfully trained with output size: 10

julia> ol.output_matrix
10×10 Matrix{Float16}:
 -1.251     3.074   -1.566     -0.10297  …   0.3823   1.341    -1.77    -0.445
  0.11017  -2.027    0.8975     0.872       -0.643    0.02615   1.083    0.615
  0.2634    3.514   -1.168     -1.532        1.486    0.1255   -1.795   -0.06555
  0.964     0.9463  -0.006855  -0.519        0.0743  -0.181    -0.433    0.06793
 -0.389     1.887   -0.702     -0.8906       0.221    1.303    -1.318    0.2634
 -0.1337   -0.4453  -0.06866    0.557    …  -0.322    0.247     0.2554   0.5933
 -0.6724    0.906   -0.547      0.697       -0.2664   0.809    -0.6836   0.2358
  0.8843   -3.664    1.615      1.417       -0.6094  -0.59      1.975    0.4785
  1.266    -0.933    0.0664    -0.4497      -0.0759  -0.03897   1.117    0.3152
  0.6353    1.327   -0.6978    -1.053        0.8037   0.6577   -0.7246   0.07336

```
"""
struct StandardRidge
    reg::Number
end

function StandardRidge(::Type{T}, reg) where {T <: Number}
    return StandardRidge(T.(reg))
end

function StandardRidge()
    return StandardRidge(0.0)
end

function train(sr::StandardRidge, states::AbstractArray, target_data::AbstractArray)
    #A = states * states' + sr.reg * I
    #b = states * target_data
    #output_layer = (A \ b)'

    if size(states, 2) != size(target_data, 2)
        throw(DimensionMismatch("\n" *
                                "\n" *
                                "  - Number of columns in `states`: $(size(states, 2))\n" *
                                "  - Number of columns in `target_data`: $(size(target_data, 2))\n" *
                                "The dimensions of `states` and `target_data` must align for training." *
                                "\n"
        ))
    end

    T = eltype(states)
    output_layer = Matrix(((states * states' + T(sr.reg) * I) \
                           (states * target_data'))')
    return OutputLayer(sr, output_layer, size(target_data, 1), target_data[:, end])
end
