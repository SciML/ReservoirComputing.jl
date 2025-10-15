
@inline function _apply_tomatrix(states_mod::F, states::AbstractMatrix) where {F <:
                                                                               Function}
    cols = axes(states, 2)
    states_1 = states_mod(states[:, first(cols)])
    new_states = similar(states_1, length(states_1), length(cols))
    new_states[:, 1] .= states_1
    for (k, j) in enumerate(cols)
        new_states[:, k] .= states_mod(@view states[:, j])
    end
    return new_states
end

"""
Pad(padding)

Padding layer that adds `padding` (either a number or an array) at the
end of a state.

## Arguments

  - `padding`: value to append. Default is 1.0.
"""
struct Pad{P} <: Function
    padding::P
end

Pad() = Pad(1.0)

function (pad::Pad)(x_old::AbstractVector)
    T = eltype(x_old)
    return vcat(x_old, T(pad.padding))
end

function (pad::Pad)(x_old::AbstractMatrix)
    T = eltype(x_old)
    row = fill(T(pad.padding), 1, size(x_old, 2))
    return vcat(x_old, row)
end

"""
Pad(padding)

Wrapper layer that concatenates the reservoir state at that
point with the input that it receives.

## Arguments

  - `op`: wrapped layer

## Examples

```julia
esn = ReservoirChain(
    Extend(
        StatefulLayer(
        ESNCell(3 => 300; init_reservoir = rand_sparse(; radius = 1.2, sparsity = 6/300))
    )
    ),
    NLAT2(),
    LinearReadout(300+3 => 3)
)
```

In this esample the input to `Extend` is the initial value fed to
[`ReservoirChain`](@ref). After `Extend`, the value in the chain will
be the state returned by the [`StatefulLayer`](@ref), `vcat`ed with
the input.
"""
@concrete struct Extend <: AbstractLuxWrapperLayer{:op}
    op <: AbstractLuxLayer
end

function initialparameters(rng::AbstractRNG, ex::Extend)
    return (op = initialparameters(rng, ex.op),)
end
function initialstates(rng::AbstractRNG, ex::Extend)
    return (op = initialstates(rng, ex.op),)
end

function (ex::Extend)(inp, ps, st::NamedTuple)
    state, st_op = apply(ex.op, inp, ps.op, st.op)
    return vcat(inp, state), (; op = st_op)
end

Base.show(io::IO, ex::Extend) = print(io, "Extend(", ex.op, ")")

@doc raw"""
    NLAT1(x)

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

julia> n_new = NLAT1(x_old)
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

julia> mat_new = NLAT1(mat_old)
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
function NLAT1(x_old::AbstractVector)
    x_new = copy(x_old)
    for idx in axes(x_old, 1)
        if isodd(idx)
            x_new[idx] = x_old[idx] * x_old[idx]
        end
    end
    return x_new
end

NLAT1(x_old::AbstractMatrix) = _apply_tomatrix(NLAT1, x_old)

NLAT1() = NLAT1

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

julia> n_new = NLAT2(x_old)
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

julia> mat_new = NLAT2(mat_old)
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
function NLAT2(x_old::AbstractVector)
    x_new = copy(x_old)
    for idx in eachindex(x_old)
        if firstindex(x_old) < idx < lastindex(x_old) && isodd(idx)
            x_new[idx, :] .= x_old[idx - 1, :] .* x_old[idx - 2, :]
        end
    end
    return x_new
end

NLAT2(x_old::AbstractMatrix) = _apply_tomatrix(NLAT2, x_old)

NLAT2() = NLAT2

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

julia> n_new = NLAT3(x_old)
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

julia> mat_new = NLAT3(mat_old)
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
function NLAT3(x_old::AbstractVector)
    x_new = copy(x_old)
    for idx in eachindex(x_old)
        if firstindex(x_old) < idx < lastindex(x_old) && isodd(idx)
            x_new[idx] = x_old[idx - 1] * x_old[idx + 1]
        end
    end
    return x_new
end

NLAT3(x_old::AbstractMatrix) = _apply_tomatrix(NLAT3, x_old)

NLAT3() = NLAT3

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
struct PartialSquare <: Function
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

(ps::PartialSquare)(x_old::AbstractMatrix) = _apply_tomatrix(ps, x_old)

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
function ExtendedSquare(x_old::AbstractVector)
    x_new = copy(x_old)
    return vcat(x_new, x_new .^ 2)
end

ExtendedSquare(x_old::AbstractMatrix) = _apply_tomatrix(ExtendedSquare, x_old)

ExtendedSquare() = ExtendedSquare
