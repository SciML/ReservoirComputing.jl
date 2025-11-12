abstract type AbstractInputEncoding end
abstract type AbstractEncodingData end

"""
    RandomMapping(permutations, expansion_size)
    RandomMapping(permutations; expansion_size=40)
    RandomMapping(;permutations=8, expansion_size=40)

Specify the **random input embedding** used by the Cellular Automata reservoir.
Each time step, the input vector of length `in_dims` is randomly placed into a
larger 1D lattice of length `expansion_size`, and this is repeated for
`permutations` independent lattices (blocks). The concatenation of these blocks
forms the CA initial condition of length: `ca_size = expansion_size * permutations`.
The detail of this implementation can be found in [Nichele2017](@cite).

## Arguments

  - `permutations`: number of independent random maps (blocks). Larger
    values increase feature diversity and `ca_size` proportionally.
  - `expansion_size`: width of each block (the size of a single CA
    lattice). Larger values increase the spatial resolution and both `ca_size`
    and `states_size`.

## Usage

This is a **configuration object**; it does not perform the mapping by itself.
Create the concrete tables with `create_encoding` and pass them to
[`RECACell`](@ref):

```julia
using ReservoirComputing, CellularAutomata, Random

in_dims = 4
generations = 8
mapping = RandomMapping(permutations = 8, expansion_size = 40)

enc = ReservoirComputing.create_encoding(mapping, in_dims, generations)  # → RandomMaps
cell = RECACell(DCA(90), enc)

rc = ReservoirChain(
    StatefulLayer(cell),
    LinearReadout(enc.states_size => in_dims; include_collect = true)
)
```

Or let [`RECA`](@ref) do this for you:

```julia
rc = RECA(in_dims = 4, out_dims = 4, DCA(90);
    input_encoding = RandomMapping(permutations = 8, expansion_size = 40),
    generations = 8)
```
"""
struct RandomMapping{I, T} <: AbstractInputEncoding
    permutations::I
    expansion_size::T
end

function RandomMapping(; permutations = 8, expansion_size = 40)
    RandomMapping(permutations, expansion_size)
end

function RandomMapping(permutations; expansion_size = 40)
    RandomMapping(permutations, expansion_size)
end

struct RandomMaps{T, E, G, M, S} <: AbstractEncodingData
    permutations::T
    expansion_size::E
    generations::G
    maps::M
    states_size::S
    ca_size::S
end

@doc raw"""
    RECACell(automaton, enc::RandomMaps)

Cellular Automata (CA)–based reservoir recurrent cell. At each time step,
the input vector is randomly embedded into a CA configuration, the CA is
evolved for a fixed number of generations, and the flattened CA evolution
is emitted as the reservoir state. The last CA configuration is carried
to the next step. For more details please refer to [Nichele2017](@cite),
and [Yilmaz2014](@cite).

## Arguments

  - `automaton`: A cellular automaton rule/object from `CellularAutomata.jl`
    (e.g., `DCA(90)`, `DCA(30)`, …).

  - `enc`: Precomputed random-mapping/encoding metadata given as a
    [`RandomMapping`](@ref).

## Inputs

  - Case A: a single input vector `x` with length
    `in_dims`. The cell internally uses the stored CA state (`st.ca`) as the
    previous configuration.

  - Case B: a tuple `(x, (ca,))` where `x` is as above and
    `ca` has length `enc.ca_size`.

## Computation

1. Random embedding of `x` into a CA initial condition `c₀` using `enc.maps`
   across `enc.permutations` blocks of length `enc.expansion_size`.

2. CA evolution for `G = enc.generations` steps with the given `automaton`,
   producing an evolution matrix `E ∈ ℝ^{(G+1) × ca_size}` where `E[1,:] = c₀`
   and `E[t+1,:] = F(E[t,:])`.

3. Feature vector is the flattened stack of `E[2:end, :]` (dropping the
   initial row), shaped as a column vector of length `enc.states_size`.

4. Carry is the final CA configuration `E[end, :]`.

## Returns

  - Output: `(h, (caₙ,))` where
      * `h` has length `enc.states_size` (the CA features),
      * `caₙ` has length `enc.ca_size` (next carry).
  - Updated (unchanged) cell state (parameters-free layer state).

## Parameters & State

  - Parameters: none
  - State: `(ca = zeros(Float32, enc.ca_size))`

"""
@concrete struct RECACell <: AbstractReservoirRecurrentCell
    automaton::Any
    enc <: RandomMaps
end

function Base.show(io::IO, reca::RECACell)
    print(io,
        "RECACell(in ⇒ ", reca.enc.ca_size, ", out=", reca.enc.states_size,
        ", gens=", reca.enc.generations, ", perms=", reca.enc.permutations,
        ", exp=", reca.enc.expansion_size, ")")
end

initialparameters(::AbstractRNG, ::RECACell) = NamedTuple()

function initialstates(::AbstractRNG, reca::RECACell)
    return (ca = zeros(Float32, reca.enc.ca_size),)
end

@doc raw"""
    RECA(in_dims, out_dims, automaton;
        input_encoding=RandomMapping(),
        generations=8, state_modifiers=(),
        readout_activation=identity)

Construct a cellular–automata reservoir model.

At each time step the input vector is randomly embedded into a Cellular
Automaton (CA) lattice, the CA is evolved for `generations` steps, and the
flattened evolution (excluding the initial row) is used as the reservoir state.
A linear [`LinearReadout`](@ref) maps these features to `out_dims`.

!!! note
    This constructor is only available when the `CellularAutomata.jl` package is
    loaded.

## Arguments

- `in_dims`: Number of input features (rows of training data).
- `out_dims`: Number of output features (rows of target data).
- `automaton`: A CA rule/object from `CellularAutomata.jl` (e.g. `DCA(90)`,
  `DCA(30)`, …).

## Keyword Arguments

- `input_encoding`: Random embedding spec with
  fields `permutations` and `expansion_size`.
  Default is `RandomMapping()`.
- `generations`: Number of CA generations to evolve per time step.
  Default is 8.
- `state_modifiers`: Optional tuple/vector of additional layers applied
  after the CA cell and before the readout (e.g., `NLAT2()`, `Pad(1.0)`,
  custom transforms, etc.). Functions are wrapped automatically.
  Default is none.
- `readout_activation`: Activation applied by the readout
  Default is `identity`.
"""
RECA(::Any...) = error("RECA requires CellularAutomata.jl; use it to enable this constructor.")
