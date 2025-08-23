function predict(rc::AbstractLuxLayer, steps::Int, ps, st; initialdata=nothing)
    if initialdata == nothing
        initialdata = rand(Float32, 3)
    end
    output = zeros(size(initialdata, 1), steps)
    for step in 1:steps
        initialdata, st = apply(rc, initialdata, ps, st)
        output[:, step] = initialdata
    end
    return output, st
end

function predict(rc::AbstractLuxLayer, data::AbstractMatrix, ps, st)
    T = size(data, 2)
    @assert T ≥ 1 "data must have at least one time step (columns)."

    y1, st = apply(rc, data[:, 1], ps, st)
    Y = similar(y1, size(y1, 1), T)
    Y[:, 1] .= y1

    @inbounds @views for t in 2:T
        yt, st = apply(rc, data[:, t], ps, st)
        Y[:, t] .= yt
    end
    return Y, st
end
