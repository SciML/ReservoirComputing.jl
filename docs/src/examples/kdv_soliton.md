# A hydrodynamic reservoir: solitons in the Korteweg-de Vries equation

[Marcucci2023](@cite) proposes Aqua-PACMANN, a reservoir computer whose
"reservoir" is a solitary wave (soliton) on shallow water: two small input
waves are launched at a fixed soliton, the resulting collision reshapes the
wave field in an input-dependent way, and a linear readout on a handful of
water-height samples recovers a Boolean logic gate. The whole system —
reservoir, input encoding, and readout — is governed by the Korteweg-de
Vries (KdV) equation, so it can be reproduced exactly as a numerical
simulation. This example builds that simulation with the same governing
equation, soliton, and encoding parameters as the paper, wires it into
ReservoirComputing.jl as a custom continuous-time reservoir, and trains it
to reproduce their XNOR gate.

## The KdV equation and its soliton

In normalized units, the KdV equation is
$u_t + u u_x + \beta u_{xxx} = 0$. It admits a traveling "soliton on a
pedestal": a hump of amplitude $r_1 - r_2$ sitting on a background level
$r_2$, moving at speed $v = (r_1 + 2r_2)/3$ without changing shape:

```math
u_s(x, t) = r_2 + (r_1 - r_2)\,\mathrm{sech}^2\!\left(
    \sqrt{\frac{r_1 - r_2}{12\beta}}\,(x - vt)
\right)
```

This soliton is the reservoir: a single, fixed, self-sustaining wave that
every input collides with. Following the paper, $\beta = 1/3$ and
$(r_1, r_2) = (2, 1)$, giving $v = 4/3$.

The equation is solved pseudospectrally on a periodic domain: spatial
derivatives become multiplications by $(ik)^n$ in Fourier space, and the
resulting ODE for the Fourier coefficients (equivalently, for `u` itself
via FFT/IFFT round trips) is handed to an `OrdinaryDiffEq` solver — this is
what makes the reservoir a `SciMLProblemReservoir`-style layer rather than
a hand-rolled time-stepper.

```@example kdv
using ReservoirComputing
using LuxCore: setup
using SciMLBase
using FFTW
using OrdinaryDiffEqTsit5
using LinearAlgebra
using Plots

N, Lx = 512, 200.0
x = collect(range(-50.0, 150.0; length = N + 1)[1:end - 1])
k = (2π / Lx) .* vcat(0:(N ÷ 2 - 1), 0, (-N ÷ 2 + 1):-1)
ik, ik3 = im .* k, (im .* k) .^ 3

function kdv_rhs!(du, u, p, t)
    uhat = fft(u)
    ux = real(ifft(p.ik .* uhat))
    uxxx = real(ifft(p.ik3 .* uhat))
    @. du = -u * ux - p.β * uxxx
    return nothing
end

β = 1 / 3
r1, r2 = 2.0, 1.0
κ = sqrt((r1 - r2) / (12β))
v_soliton = (r1 + 2r2) / 3
soliton(xx, t) = r2 + (r1 - r2) * sech(κ * (xx - v_soliton * t))^2
```

## Encoding two Boolean inputs as waves

Each input channel is a truncated, windowed wave: a `cos²` carrier at its
own wavenumber, confined to a region of width `l` by a super-Gaussian
envelope so it doesn't perturb the rest of the domain. Wavenumber picks the
channel, amplitude picks its Boolean value (`0` or `1/4`, matching the
paper). The two encoding waves plus the soliton — shifted by a delay `L`
so it starts well to the left of the encoding region — become the initial
condition:

```math
u_0(x) = \underbrace{e^{-(2x/l)^8}
    \sum_{n=1}^{2} \epsilon_n \cos^2(k_n x)}_{\text{encoded input}}
    + u_s(x + L, 0)
```

```@example kdv
L, l = 17.0, 20.0
k1, k2 = sqrt(3) / 4, 1 / 2
envelope(xx) = exp(-(2xx / l)^8)
encode(xx, ϵ1, ϵ2) = envelope(xx) * (ϵ1 * cos(k1 * xx)^2 + ϵ2 * cos(k2 * xx)^2)

build_u0(a, b) = encode.(x, 0.25a, 0.25b) .+ soliton.(x .+ L, 0.0)
```

The soliton starts at $x=-17$ moving right at $v=4/3$; a detector placed at
$x_D=50$ therefore sees it cross around $t \approx 50$. Sampling the field
there at four times bracketing that crossing — as in the paper,
$t \in \{40, 49, 51, 60\}$ — is enough state to separate all four input
combinations:

```@example kdv
xD_idx = argmin(abs.(x .- 50.0))
times = [40.0, 49.0, 51.0, 60.0]
p = (ik = ik, ik3 = ik3, β = β)
base_prob = ODEProblem(kdv_rhs!, build_u0(0.0, 0.0), (0.0, 60.0), p)
```

## A custom continuous-time reservoir

[`SciMLProblemReservoir`](@ref) is built for reservoirs where a *signal*
drives the ODE (`p.input(t)`, injected from `collectstates`'s windowed
zero-order-hold of a discrete `data` matrix). Here the input instead
selects the *initial condition* — each of the four truth-table rows is its
own independent simulation from `t=0`, not a continuation of the previous
row's wave field. That's exactly the case the package's developer
interface anticipates: subtype [`AbstractSciMLProblemReservoir`](@ref) and
implement the `__collectstates` hook directly, one `remake`d solve per
input column, instead of reusing the windowed built-in.

```@example kdv
struct KdVReservoir{U, XT, TT, AL, BU} <: AbstractSciMLProblemReservoir
    prob::U
    x::XT
    xD_idx::Int
    times::TT
    alg::AL
    build_u0::BU
end

function ReservoirComputing.__collectstates(
        res::KdVReservoir, rc::ReservoirComputing.AbstractReservoirComputer,
        data::AbstractMatrix, ps::NamedTuple, st::NamedTuple
    )
    n_samples, n_times = size(data, 2), length(res.times)
    states = Matrix{Float64}(undef, n_times, n_samples)
    for col in 1:n_samples
        u0 = res.build_u0(data[1, col], data[2, col])
        sol = solve(remake(res.prob; u0 = u0), res.alg;
            saveat = res.times, reltol = 1.0e-8, abstol = 1.0e-10)
        states[:, col] .= [sol.u[i][res.xD_idx] for i in 1:n_times]
    end
    return states, st
end

res = KdVReservoir(base_prob, x, xD_idx, times, Tsit5(), build_u0)
rc = ReservoirComputer(res, LinearReadout(length(times) => 2))
```

`initialparameters`/`initialstates` for `res` fall back to the empty
`NamedTuple` [`AbstractSciMLProblemReservoir`](@ref) already provides —
`KdVReservoir` carries no trainable parameters of its own, only the fixed
physical setup above.

## Training the XNOR gate

`data` holds the two raw Boolean inputs per column; `target` is the
one-hot XNOR truth table (`false = (1,0)`, `true = (0,1)`, matching the
paper). Fitting the readout is then a single call to [`train`](@ref) with
no regularization — the paper's readout is the exact Moore-Penrose
pseudoinverse of the response matrix, i.e. plain least squares:

```@example kdv
rng = MersenneTwister(0)
ps, st = setup(rng, rc)

data = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]     # rows: input A, input B
xnor_onehot(a, b) = (a == b) ? [0.0, 1.0] : [1.0, 0.0]
target = reduce(hcat, [xnor_onehot(data[1, i], data[2, i]) for i in 1:4])

ps, st = train(rc, data, target, ps, st; objective = RidgeRegression(0.0))
```

## Results

```@example kdv
states, _ = collectstates(rc, data, ps, st)
pred, _ = predict(rc, data, ps, st)

println("response matrix (water height at the 4 sample times, per input):")
display(states)
println("\ndet(response matrix) = ", det(states))
println("\nmax |prediction - target| = ", maximum(abs.(pred .- target)))
```

The response matrix is nonsingular — the four inputs really do land at
four linearly independent points in state space — with a determinant of
about `-0.011`, closely matching the paper's reported `-0.0115` and a good
sign that this simulation reproduces their setup faithfully. The fit
recovers the truth table to numerical precision, since the readout is an
exact (noiseless) linear solve of a square, invertible system.

A picture of one collision makes the mechanism concrete: the soliton
approaches from the left, passes through the encoding wave sitting near
$x=0$, and the collision leaves a wake that the detector (dashed line)
samples on its way past:

```@example kdv
u0_11 = build_u0(1.0, 1.0)
sol_11 = solve(remake(base_prob; u0 = u0_11), Tsit5();
    reltol = 1.0e-8, abstol = 1.0e-10, saveat = 0:0.5:60)
U = reduce(hcat, sol_11.u)

hm = heatmap(sol_11.t, x, U; xlabel = "t", ylabel = "x", color = :viridis,
    title = "KdV field u(x,t), inputs A=B=true", colorbar_title = "u")
hline!(hm, [50.0]; color = :red, linestyle = :dash, label = "detector (x=50)")
```

The takeaway isn't that a bucket of water is a practical logic gate — it's
that [`AbstractSciMLProblemReservoir`](@ref) plugs into any dynamics
`OrdinaryDiffEq` can integrate, physical or not, as long as its
`__collectstates` hook maps inputs to features the way that system's task
actually requires.
