# Migrating to ReservoirComputing.jl v1

Changes from the last `0.12` releases to v1. Deprecated names still emit
warnings on `0.12.x` and are removed at the v1.0 cut.

## Initializer sign patterns

Initializer sign handling now uses typed sign-pattern objects instead of resolving a
function from the `sampling_type` symbol and forwarding sampler-specific keywords.

The available patterns are:

- `RandomSigns(positive_probability)`: independently preserves each existing sign with
  the given probability and flips it otherwise. The default probability is `0.5`.
- `RegularSigns(strides)`: flips signs at positions selected by an integer stride or a
  repeating tuple of strides. Vectors of strides are also accepted.
- `IrrationalDigitSigns(irrational; start)`: flips signs where the corresponding decimal
  digit of an irrational number is odd.
- `nothing`: leaves every sign unchanged. This is the default.

Pass the pattern through the `signs` keyword:

```julia
minimal_init(100, 3; signs = RandomSigns(0.5))
minimal_init(100, 3; signs = RegularSigns(2))
minimal_init(100, 3; signs = RegularSigns((2, 3)))
minimal_init(100, 3; signs = IrrationalDigitSigns(pi; start = 1))
minimal_init(100, 3) # signs = nothing
```

### Migrating `sampling_type`

`sampling_type` and its sampler-specific forwarded keywords are removed in v1.0.

| Before v1 | v1 replacement |
|:----------|:---------------|
| `sampling_type = :no_sample` | `signs = nothing` |
| `sampling_type = :bernoulli_sample!, positive_prob = p` | `signs = RandomSigns(p)` |
| `sampling_type = :regular_sample!, strides = s` | `signs = RegularSigns(s)` |
| `sampling_type = :irrational_sample!` | `signs = IrrationalDigitSigns(x; start = n)` |

Do not pass `signs` together with `sampling_type`. Sampler-specific keywords such as
`positive_prob`, `strides`, and `irrational` now belong to the corresponding pattern
constructor instead of the initializer.

This migration applies to the component-building functions and to initializers that
forward their keyword arguments to those components, including:

- `delay_line!` and `delay_line`;
- `backward_connection!` and reservoir initializers containing backward connections;
- `simple_cycle!`, `reverse_simple_cycle!`, and cycle-based reservoir initializers;
- `add_jumps!` and `cycle_jumps`;
- `self_loop!` and self-loop reservoir initializers;
- `minimal_init` and `weighted_minimal`.

Nested keyword bundles accept sign patterns directly:

```julia
cycle_jumps(100, 100; jump_kwargs = (; signs = RegularSigns((2, 3))))
```

### `minimal_init` default

`minimal_init` previously applied Bernoulli sign flipping by default. Its new default is
`signs = nothing`, so generated weights retain their original signs. Pass
`signs = RandomSigns()` to preserve the previous default explicitly:

```julia
minimal_init(100, 3; signs = RandomSigns())
```

## Initializer behavior corrections

`informed_init` now extracts scalar random values correctly when assigning informed
input connections. Earlier versions could fail on zero-dimensional array arithmetic.
Call sites do not need changes.

## Input-extended reservoir states

`Extend` can be passed in `state_modifiers` to prepend the current model input to the
wrapped modifier's output:

```julia
model = ESN(3, 100, 3; state_modifiers = (Extend(Collect()),))
```

High-level constructors size their linear readout automatically for `Extend` when
its wrapped operation preserves the feature width. For a custom modifier that changes
the feature width, pass the resulting width explicitly through `readout_in_dims`.

## Training API

```julia
ps, st = train(model, train_data, target_data, ps, st;
    objective = RidgeRegression(1e-3))
```

### Generic `train!` → `train`

The model-level bang API is removed in v1.0. Map the positional training method to
`objective`:

| Before v1 | v1 replacement |
|:----------|:---------------|
| `train!(model, data, targets, ps, st, StandardRidge(1e-3))` | `train(model, data, targets, ps, st; objective = RidgeRegression(1e-3))` |
| `train!(model, data, targets, ps, st)` | `train(model, data, targets, ps, st)` |

`washout`, `return_states`, and `solver` keep the same names. Conceptor
`train!(rng, concept::Conceptor, ...)` is unchanged.

### Objective-level `train`

`train(objective, states, targets; solver=...)` is removed in v1.0. Fit through the
model instead (returns `(ps, st)`, not a bare weight matrix):

```julia
# before
W = train(RidgeRegression(1e-3), states, targets)

# after
ps, st = train(model, train_data, target_data, ps, st;
    objective = RidgeRegression(1e-3))
```

## Renames

| Before v1 | v1 |
|:----------|:---|
| `StandardRidge` | [`RidgeRegression`](@ref) |
| `toepliz_init` | [`toeplitz_init`](@ref) |
| `model.states_modifiers` | `model.state_modifiers` |

## Conceptor `tolerance`

[`conceptor_and`](@ref) accepts optional `tolerance` for its relative rank cutoff.
Default `nothing` preserves the previous behavior:

```julia
conceptor_and(C1, C2; tolerance = 1e-10)
```
