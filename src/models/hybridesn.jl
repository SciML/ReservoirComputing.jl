############################
# Knowledge-model wrapper  #
############################

struct KnowledgeModel{T,K,O,I,S,D}
    prior_model::T
    u0::K
    tspan::O
    dt::I
    datasize::S
    model_data::D
end

"""
    KnowledgeModel(prior_model, u0, tspan, datasize)

Build a `KnowledgeModel` and precompute `model_data` on a time grid of length
`datasize+1`. The extra step aligns with teacher-forced (xₜ → yₜ₊₁) usage.
"""
function KnowledgeModel(prior_model, u0, tspan, datasize)
    trange = collect(range(tspan[1], tspan[2]; length=datasize))
    @assert length(trange) ≥ 2 "datasize must be ≥ 2 to infer dt"
    dt = trange[2] - trange[1]
    tsteps = push!(trange, trange[end] + dt)
    tspan2 = (tspan[1], tspan[2] + dt)
    mdl = prior_model(u0, tspan2, tsteps)
    return KnowledgeModel(prior_model, u0, tspan, dt, datasize, mdl)
end

# Helper: forecast a KB stream for `steps` auto-regressive steps beyond tspan
function kb_forecast(km::KnowledgeModel, steps::Integer)
    @assert steps ≥ 1
    t0 = km.tspan[2] + km.dt
    tgrid = collect(t0:km.dt:(t0+km.dt*(steps-1)))
    tspan = (t0, tgrid[end])
    u0 = km.model_data[:, end]
    mdl = km.prior_model(u0, tspan, [t0; tgrid[2:end]])
    return mdl
end

kb_stream_train(km::KnowledgeModel, T::Integer) = km.model_data[:, 1:T]


# Concats a column from `stream` at each step:  z_t = vcat(x_t, stream[:, i])
@concrete struct AttachStream <: AbstractLuxLayer
    stream <: AbstractMatrix
end

initialparameters(::AbstractRNG, ::AttachStream) = NamedTuple()
initialstates(::AbstractRNG, ::AttachStream) = (i=1,)

function (l::AttachStream)(x::AbstractVector, ps, st::NamedTuple)
    @boundscheck (st.i ≤ size(l.stream, 2)) ||
                 throw(BoundsError(l.stream, st.i))
    out = vcat(x, @view l.stream[:, st.i])
    return out, (i=st.i + 1,)
end

"""
    HybridESN(km::KnowledgeModel,
              in_dims::Integer, res_dims::Integer, out_dims::Integer,
              activation=tanh;
              state_modifiers=(),
              readout_activation=identity,
              include_collect=true,
              kwargs...)

Build a hybrid ESN as a `ReservoirChain`:
`StatefulLayer(ESNCell) → modifiers → AttachStream(train KB) → LinearReadout`.
"""
function HybridESN(km::KnowledgeModel,
    in_dims::Integer, res_dims::Integer, out_dims::Integer,
    activation=tanh;
    state_modifiers=(),
    readout_activation=identity,
    include_collect::Bool=true,
    kwargs...)
    cell = ESNCell(in_dims => res_dims, activation; kwargs...)

    mods = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
           Tuple(state_modifiers) : (state_modifiers,)
    stream_train = kb_stream_train(km, km.datasize)
    d_kb = size(stream_train, 1)

    ro = LinearReadout((res_dims + d_kb) => out_dims, readout_activation;
        include_collect=static(include_collect))

    return ReservoirChain((StatefulLayer(cell), mods..., AttachStream(stream_train), ro)...)
end

function with_kb_stream(rc::ReservoirChain, new_stream::AbstractMatrix)
    layers = rc.layers
    names = propertynames(layers)
    vals = collect(Tuple(layers))
    found = false
    for (k, v) in enumerate(vals)
        if v isa AttachStream
            vals[k] = AttachStream(new_stream)
            found = true
            break
        end
    end
    @assert found "No AttachStream layer found in chain."
    new_nt = NamedTuple{names}(Tuple(vals))
    return ReservoirChain(new_nt, rc.name)
end
