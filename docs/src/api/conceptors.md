# Conceptors

Conceptors after Jaeger (2014), [Jaeger2014conceptors](@cite). See the
[Morphing patterns with conceptors](@ref) example for an end-to-end walkthrough.

## Conceptor matrices

```@docs
    conceptor_matrix
    correlation_matrix
    conceptor_from_states
    conceptor_singular_values
    quota
```

## Aperture adaptation

```@docs
    aperture_adapt
    adapt_singular_value
    reaperture
    attenuation
    optimal_aperture
```

## Boolean algebra

```@docs
    conceptor_not
    conceptor_and
    conceptor_or
```

## The `Conceptor` wrapper and its library

```@docs
    Conceptor
    has_conceptor
    get_conceptor
    store_conceptor
    set_active_conceptor
    active_conceptor
```

## Loading, generation, and morphing

```@docs
    load
    generate
    morph_conceptor
```

## Conceptor-filtered training

```@docs
    store_conceptors
```
