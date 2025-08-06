abstract type AbstractStates end
abstract type AbstractPaddedStates <: AbstractStates end
abstract type NonLinearAlgorithm end

function pad_state!(states_type::AbstractPaddedStates, x_pad, x)
    x_pad[1, :] .= states_type.padding
    x_pad[2:end, :] .= x
    return x_pad
end

function pad_state!(states_type, x_pad, x)
    return x
end

#states types
"""
    StandardStates()

When this struct is employed, the states of the reservoir are not modified.

# Example

```jldoctest
julia> states = StandardStates()
StandardStates()

julia> test_vec = zeros(Float32, 5)
5-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0

julia> new_vec = states(test_vec)
5-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0

julia> test_mat = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> new_mat = states(test_mat)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
struct StandardStates <: AbstractStates end

function (::StandardStates)(nla_type::NonLinearAlgorithm,
        state, inp)
    return nla(nla_type, state)
end

(::StandardStates)(state) = state
"""
    ExtendedStates()

The `ExtendedStates` struct is used to extend the reservoir
states by vertically concatenating the input data (during training)
and the prediction data (during the prediction phase).

# Example

```jldoctest
julia> states = ExtendedStates()
ExtendedStates()

julia> test_vec = zeros(Float32, 5)
5-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0

julia> new_vec = states(test_vec, fill(3.0f0, 3))
8-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0
 3.0
 3.0
 3.0

julia> test_mat = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> new_mat = states(test_mat, fill(3.0f0, 3))
8×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 3.0  3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0  3.0
```
"""
struct ExtendedStates <: AbstractStates end

function (states_type::ExtendedStates)(mat::AbstractMatrix, inp::AbstractMatrix)
    results = states_type.(eachcol(mat), eachcol(inp))
    return hcat(results...)
end

function (states_type::ExtendedStates)(mat::AbstractMatrix, inp::AbstractVector)
    results = Vector{Vector{eltype(mat)}}(undef, size(mat, 2))
    for (idx, col) in enumerate(eachcol(mat))
        results[idx] = states_type(col, inp)
    end
    return hcat(results...)
end

function (::ExtendedStates)(vect::AbstractVector, inp::AbstractVector)
    return x_tmp = vcat(vect, inp)
end

function (states_type::ExtendedStates)(nla_type::NonLinearAlgorithm,
        state::AbstractVecOrMat, inp::AbstractVecOrMat)
    return nla(nla_type, states_type(state, inp))
end

"""
    PaddedStates(padding)
    PaddedStates(;padding=1.0)

Creates an instance of the `PaddedStates` struct with specified
padding value (default 1.0). The states of the reservoir are padded
by vertically concatenating the padding value.

# Example

```jldoctest
julia> states = PaddedStates(1.0)
PaddedStates{Float64}(1.0)

julia> test_vec = zeros(Float32, 5)
5-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0

julia> new_vec = states(test_vec)
6-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0
 1.0

julia> test_mat = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> new_mat = states(test_mat)
6×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0
```
"""
struct PaddedStates{T} <: AbstractPaddedStates
    padding::T
end

function PaddedStates(; padding = 1.0)
    return PaddedStates(padding)
end

function (states_type::PaddedStates)(mat::AbstractMatrix)
    results = states_type.(eachcol(mat))
    return hcat(results...)
end

function (states_type::PaddedStates)(vect::AbstractVector)
    tt = eltype(vect)
    return vcat(vect, tt(states_type.padding))
end

function (states_type::PaddedStates)(nla_type::NonLinearAlgorithm,
        state::AbstractVecOrMat, inp::AbstractVecOrMat)
    return nla(nla_type, states_type(state))
end

"""
    PaddedExtendedStates(padding)
    PaddedExtendedStates(;padding=1.0)

Constructs a `PaddedExtendedStates` struct, which first extends
the reservoir states with training or prediction data,then pads them
with a specified value (defaulting to 1.0).

# Example

```jldoctest
julia> states = PaddedExtendedStates(1.0)
PaddedExtendedStates{Float64}(1.0)

julia> test_vec = zeros(Float32, 5)
5-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0

julia> new_vec = states(test_vec, fill(3.0f0, 3))
9-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.0
 0.0
 1.0
 3.0
 3.0
 3.0

julia> test_mat = zeros(Float32, 5, 5)
5×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> new_mat = states(test_mat, fill(3.0f0, 3))
9×5 Matrix{Float32}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0
 3.0  3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0  3.0
```
"""
struct PaddedExtendedStates{T} <: AbstractPaddedStates
    padding::T
end

function PaddedExtendedStates(; padding = 1.0)
    return PaddedExtendedStates(padding)
end

function (states_type::PaddedExtendedStates)(nla_type::NonLinearAlgorithm,
        state::AbstractVecOrMat, inp::AbstractVecOrMat)
    return nla(nla_type, states_type(state, inp))
end

function (states_type::PaddedExtendedStates)(state::AbstractVecOrMat,
        inp::AbstractVecOrMat)
    x_pad = PaddedStates(states_type.padding)(state)
    x_ext = ExtendedStates()(x_pad, inp)
    return x_ext
end

#### non linear algorithms ###
## to conform to current (0.10.5) approach
nla(nlat::NonLinearAlgorithm, x_old::AbstractVecOrMat) = nlat(x_old)

# dispatch over matrices for all nonlin algorithms
function (nlat::NonLinearAlgorithm)(x_old::AbstractMatrix)
    results = nlat.(eachcol(x_old))
    return hcat(results...)
end

"""
    NLADefault()

`NLADefault` represents the default non-linear algorithm option.
When used, it leaves the input array unchanged.

# Example

```jldoctest
julia> nlat = NLADefault()
NLADefault()

julia> x_old = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
10-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9

julia> n_new = nlat(x_old)
10-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9

julia> mat_old = [1 2 3;
                  4 5 6;
                  7 8 9;
                  10 11 12;
                  13 14 15;
                  16 17 18;
                  19 20 21]
7×3 Matrix{Int64}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
 13  14  15
 16  17  18
 19  20  21

julia> mat_new = nlat(mat_old)
7×3 Matrix{Int64}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
 13  14  15
 16  17  18
 19  20  21
```
"""
struct NLADefault <: NonLinearAlgorithm end

(::NLADefault)(x::AbstractVector) = x
(::NLADefault)(x::AbstractMatrix) = x

@doc raw"""
    NLAT1()

`NLAT1` implements the T₁ transformation algorithm introduced
in [Chattopadhyay2020](@cite) and [Pathak2017](@cite). The T₁ algorithm squares
elements of the input array, targeting every second row.


```math
\tilde{r}_{i,j} =
\begin{cases}
    r_{i,j} \times r_{i,j}, & \text{if } j \text{ is odd}; \\
    r_{i,j}, & \text{if } j \text{ is even}.
\end{cases}
```
# Example

```jldoctest
julia> nlat = NLAT1()
NLAT1()

julia> x_old = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
10-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9

julia> n_new = nlat(x_old)
10-element Vector{Int64}:
  0
  1
  4
  3
 16
  5
 36
  7
 64
  9

julia> mat_old = [1  2  3;
                   4  5  6;
                   7  8  9;
                  10 11 12;
                  13 14 15;
                  16 17 18;
                  19 20 21]
7×3 Matrix{Int64}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
 13  14  15
 16  17  18
 19  20  21

julia> mat_new = nlat(mat_old)
7×3 Matrix{Int64}:
   1    4    9
   4    5    6
  49   64   81
  10   11   12
 169  196  225
  16   17   18
 361  400  441

```
"""
struct NLAT1 <: NonLinearAlgorithm end

function (::NLAT1)(x_old::AbstractVector)
    x_new = copy(x_old)

    for idx in eachindex(x_old)
        if isodd(idx)
            x_new[idx] = x_old[idx] * x_old[idx]
        end
    end

    return x_new
end

@doc raw"""
    NLAT2()

`NLAT2` implements the T₂ transformation algorithm as defined
in [Chattopadhyay2020](@cite). This transformation algorithm modifies the
reservoir states by multiplying each odd-indexed row
(starting from the second row) with the product of its two preceding rows.

```math
\tilde{r}_{i,j} =
\begin{cases}
    r_{i,j-1} \times r_{i,j-2}, & \text{if } j > 1 \text{ is odd}; \\
    r_{i,j}, & \text{if } j \text{ is 1 or even}.
\end{cases}
```
# Example

```jldoctest
julia> nlat = NLAT2()
NLAT2()

julia> x_old = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
10-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9

julia> n_new = nlat(x_old)
10-element Vector{Int64}:
  0
  1
  0
  3
  6
  5
 20
  7
 42
  9

julia> mat_old = [1  2  3;
                   4  5  6;
                   7  8  9;
                  10 11 12;
                  13 14 15;
                  16 17 18;
                  19 20 21]
7×3 Matrix{Int64}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
 13  14  15
 16  17  18
 19  20  21

julia> mat_new = nlat(mat_old)
7×3 Matrix{Int64}:
  1   2    3
  4   5    6
  4  10   18
 10  11   12
 70  88  108
 16  17   18
 19  20   21

```
"""
struct NLAT2 <: NonLinearAlgorithm end

function (::NLAT2)(x_old::AbstractVector)
    x_new = copy(x_old)

    for idx in eachindex(x_old)
        if firstindex(x_old) < idx < lastindex(x_old) && isodd(idx)
            x_new[idx, :] .= x_old[idx - 1, :] .* x_old[idx - 2, :]
        end
    end

    return x_new
end

@doc raw"""
    NLAT3()

Implements the T₃ transformation algorithm as detailed
in [Chattopadhyay2020](@cite). This algorithm modifies the reservoir's states by
multiplying each odd-indexed row (beginning from the second row) with the
product of the immediately preceding and the immediately following rows.

```math
\tilde{r}_{i,j} =
\begin{cases}
r_{i,j-1} \times r_{i,j+1}, & \text{if } j > 1 \text{ is odd}; \\
r_{i,j}, & \text{if } j = 1 \text{ or even.}
\end{cases}
```
# Example

```jldoctest
julia> nlat = NLAT3()
NLAT3()

julia> x_old = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
10-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9

julia> n_new = nlat(x_old)
10-element Vector{Int64}:
  0
  1
  3
  3
 15
  5
 35
  7
 63
  9

julia> mat_old = [1  2  3;
                   4  5  6;
                   7  8  9;
                  10 11 12;
                  13 14 15;
                  16 17 18;
                  19 20 21]
7×3 Matrix{Int64}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
 13  14  15
 16  17  18
 19  20  21

julia> mat_new = nlat(mat_old)
7×3 Matrix{Int64}:
   1    2    3
   4    5    6
  40   55   72
  10   11   12
 160  187  216
  16   17   18
  19   20   21

```
"""
struct NLAT3 <: NonLinearAlgorithm end

function (::NLAT3)(x_old::AbstractVector)
    x_new = copy(x_old)

    for idx in eachindex(x_old)
        if firstindex(x_old) < idx < lastindex(x_old) && isodd(idx)
            x_new[idx] = x_old[idx - 1] * x_old[idx + 1]
        end
    end

    return x_new
end

@doc raw"""
    PartialSquare(eta)

Implement a partial squaring of the states as described in [Barbosa2021](@cite).

# Equations

```math
    \begin{equation}
    g(r_i) =
    \begin{cases} 
        r_i^2, & \text{if } i \leq \eta_r N, \\
        r_i, & \text{if } i > \eta_r N.
    \end{cases}
    \end{equation}
```

# Examples

```jldoctest
julia> ps = PartialSquare(0.6)
PartialSquare(0.6)


julia> x_old = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
10-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9

julia> x_new = ps(x_old)
10-element Vector{Int64}:
  0
  1
  4
  9
 16
 25
  6
  7
  8
  9
```
"""
struct PartialSquare <: NonLinearAlgorithm
    eta::Number
end

function (ps::PartialSquare)(x_old::AbstractVector)
    x_new = copy(x_old)
    n_length = length(x_old)
    threshold = floor(Int, ps.eta * n_length)
    for idx in eachindex(x_old)
        if idx <= threshold
            x_new[idx] = x_old[idx]^2
        end
    end

    return x_new
end

@doc raw"""

    ExtendedSquare()

Extension of the Lu initialization proposed in [Herteux2020](@cite).
The state vector is extended with the squared elements of the initial
state

# Equations

```math
\begin{equation}
    \vec{x} = \{x_1, x_2, \dots, x_N, x_1^2, x_2^2, \dots, x_N^2\}
\end{equation}
```

# Examples

```jldoctest
julia> x_old = [1, 2, 3, 4, 5, 6, 7, 8, 9]
9-element Vector{Int64}:
 1
 2
 3
 4
 5
 6
 7
 8
 9

julia> es = ExtendedSquare()
ExtendedSquare()

julia> x_new = es(x_old)
18-element Vector{Int64}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
  1
  4
  9
 16
 25
 36
 49
 64
 81

```
"""
struct ExtendedSquare <: NonLinearAlgorithm end

function (::ExtendedSquare)(x_old::AbstractVector)
    x_new = copy(x_old)
    return vcat(x_new, x_new .^ 2)
end
