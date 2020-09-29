module Drifter
using StaticArrays
using LinearAlgebra: dot, norm
using Transducers
using Optim

export drift_estimation

const P2D = SVector{2,Float64}

include("celllists.jl")

struct FrameToFrameShift
    N_matches :: Int64
    average_shift :: P2D
end

struct Frame
    points :: Vector{P2D}
    drift :: P2D
    celllist :: CellList
end

function compute_ave_offset(f1 :: Frame, f2 :: Frame)
    sum_offsets = SVector(0.0,0.0)
    num_matches = 0
    for p in f1.points
        flag, n = neighbor(f2.celllist, p - f1.drift + f2.drift)
        if flag
            sum_offsets += n - p
            num_matches += 1
        end
    end
    FrameToFrameShift(num_matches, sum_offsets/num_matches)
end

function indexpairs(n_frames, offsets)
    ((i,o) for i in 1:n_frames-1, o in offsets) |> Map(((i,o),) -> (i,i+o)) |>
        Filter(((i,j),) -> 1 <= j <= n_frames) |> collect
end

function cached_link(old_value, i,j, frames, min_dist_cache, old_drift, new_drift)
    if norm((old_drift[i] - old_drift[j]) - (new_drift[i]-new_drift[j])) < min_dist_cache
        (i,j,old_value)
    else
        (i,j,compute_ave_offset(frames[i], frames[j]))
    end
end

struct DriftLoss
    drift_pairs :: Vector{Tuple{Int64,Int64, FrameToFrameShift}}
    lambda :: Float64
end

#TODO: Find a more elegant way to handle x[1] = 0.0...
function loss_and_gradient!(g, l :: DriftLoss, x :: Vector{Float64}, :: Val{I}) where {I}
    fill!(g, 0.0)
    r = 0.0
    for (f_i,f_j, d) in l.drift_pairs
        if d.N_matches > 0
            if f_i != 1
                est_drift = x[f_j-1] - x[f_i-1]
                residual = est_drift-d.average_shift[I]
                r += (d.N_matches/2.0)*residual^2
                g[f_j-1] += d.N_matches*residual
                g[f_i-1] -= d.N_matches*residual
            else
                est_drift = x[f_j-1] - 0.0
                residual = est_drift-d.average_shift[I]
                r += (d.N_matches/2.0)*residual^2
                g[f_j-1] += d.N_matches*residual
            end
        end
    end

    # discrete second derivative of drift...
    #first term...
    r += (l.lambda/2)*(-2x[1] + x[2])^2
    g[1] += -2*l.lambda*(-2x[1] + x[2])
    g[2] += l.lambda*(-2x[1] + x[2])
    for i in 2:length(x)-1
        residual = (x[i-1] - 2x[i] + x[i+1])
        r += (l.lambda/2)*residual^2
        g[i-1] += l.lambda*residual
        g[i] -= 2*l.lambda*residual
        g[i+1] += l.lambda*residual
    end
    r
end

function minimize_wrt_drift(l, x_init = zeros(2*l.frame_max))
    #TODO: Use a preconditioner here? Especially when lambda is large...
    res_x = Optim.optimize(Optim.only_fg!((F,G,x)->loss_and_gradient!(G, l, x, Val{1}())), getindex.(x_init,1)[2:end], method = Optim.LBFGS(m=17), g_tol=1E-10)
    res_y = Optim.optimize(Optim.only_fg!((F,G,x)->loss_and_gradient!(G, l, x, Val{2}())), getindex.(x_init,2)[2:end], method = Optim.LBFGS(m=17), g_tol=1E-10)
    pushfirst!([P2D(x,y) for (x,y) in zip(res_x.minimizer,res_y.minimizer)], P2D(0.0,0.0))
end


"""
Estimate the 2D drift for the vector of vectors of points in `localizations`
(each vector of points in `localizations` is the localizations in a single frame).

`frame_offsets :: Vector{Int64}` contains the offsets `o` s.t. `localizations[i]` should be compared with `localizations[i+o]` when
estimating the drift. Generally speaking more offsets will make the estimate more accurate
but also more computationally expensive. If accurate long-range drift is required be sure to include some larger offsets, e.g.
`frame_offsets = vcat(1:50, 10:5:100, 125:50:4000)`.

It may be substantially faster to ~warmstart~ the optimization (using the keyword argument `drift`) and use a series of increasingly large `frame_offset` vectors.

`max_dist` is the maximum distance between two localizations in different frames that will be allowed before they are
considered to be localizations of different objects. Usually 1-3x the localization precision works here (depending on the density of the data).

The keywork argument `lambda` is included for regularization - the objective function is something like

drift ↦ OT(drift, offsets) + λ * ||D*drift||^2

where D is the discrete second derivative.
"""
function drift_estimation(localizations :: Vector{Vector{P2D}}, frame_offsets :: Vector{Int64}, max_dist, max_iters; drift = [P2D(0.0, 0.0) for _ in 1:length(localizations)], lambda = 0.0, recompute_max_dist = 1E-3)
    frames = [Frame(locs, d, CellList(locs, max_dist)) for (locs, d) in zip(localizations, drift)]
    cache = indexpairs(length(localizations), frame_offsets) |> Map(((i,j),) -> (i,j,compute_ave_offset(frames[i], frames[j]))) |> tcollect

    old_drift = drift

    for i in 1:max_iters
        delta_drift = sum(sum(abs2, d - nd) for (d, nd) in zip(drift, old_drift)) / length(drift)

        if i != 1 && delta_drift <= 1E-7
            break
        end

        #update frames
        frames = [Frame(d.points, drift, d.celllist) for (d, drift) in zip(frames, drift)]

        # link across frames (~ minimize w.r.t matching)
        cache = cache |> Map(((i,j,v),) -> cached_link(v, i,j,frames, recompute_max_dist, old_drift, drift))  |> tcollect
        # minimize loss w.r.t drift
        old_drift = drift
        drift = minimize_wrt_drift(DriftLoss(cache,lambda), drift)
    end
    [[z - d for z in f] for (f,d) in zip(localizations,drift)], drift
end

end