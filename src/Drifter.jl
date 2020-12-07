module Drifter
using SparseArrays
using LinearAlgebra

export drift_estimation
export CellList, neighbor, neighbors_fold

include("celllists.jl")

#TODO: Make this banded. Or just work it out by hand.
function _discrete_first_derivative(n)
    I = Int64[]
    J = Int64[]
    V = Float64[]
    sizehint!(I, 2*(n-1))
    sizehint!(J, 2*(n-1))
    sizehint!(V, 2*(n-1))
    for i in 1:n-1
        push!(I, i)
        push!(J, i)
        push!(V, 1.0)
        push!(I, i)
        push!(J, i+1)
        push!(V, -1.0)
    end
    sparse(I,J,V)
end

function drift_estimation(localizations :: Vector{Vector{NTuple{2, Float64}}}, max_dist, max_iters; drift = fill((0.0,0.0), length(localizations)), λ = 0.0, ftol_rel = 1E-6, max_frames = length(drift))
        old_matches = 0
        R = _discrete_first_derivative(length(localizations))
        K_R = λ*(R'R)[2:end, 2:end];

        for i in 1:max_iters
            frames = [[(p.-d,i) for p in ps] for (i,(d,ps)) in enumerate(zip(drift,localizations))]
            framed_points = vcat(frames...)

            println()
            @time cl = CellList(framed_points, max_dist, -1)
            
            println("Form normal equations")
            @time  (K_x,b_x,K_y,b_y) = _build_normal_equations(frames, cl, max_dist, max_frames)
            matches = div(sum(Int64, diag(K_x)), 2)
            @show i, matches
            f_rel = (matches - old_matches)/matches
            @show f_rel
            old_matches = matches
            println("Solve")

            t_x = (b_x[2:end] - (K_R*(first.(drift)[2:end])))
            t_y = (b_y[2:end] - (K_R*(last.(drift)[2:end])))
            
            @time d_x = vcat(0.0,(K_x[2:end,2:end] + K_R)\t_x)
            @time d_y = vcat(0.0,(K_y[2:end,2:end] + K_R)\t_y)
            drift = [d .+ (x,y) for (d,x,y) in zip(drift,d_x, d_y)]
            if f_rel < ftol_rel
                break
            end
        end    
    [[p .- d for p in ps] for (ps,d) in zip(localizations, drift)], drift
end
    
struct NE
    storage :: Vector{NTuple{4, Float64}} # K_D, K_r, b_x, b_y
end
    
NE(n :: Int64) = NE(fill((0.0,0.0,0.0,0.0), n))

function zero!(s :: NE)
    @inbounds for i in 1:length(s.storage)
        s.storage[i] = (0.0,0.0,0.0,0.0)
    end
end

@inline function _update_ne((s, p, f_i, r_sq, max_frames), (n, f_j))
    if f_i < f_j && f_j - f_i < max_frames && _squared_dist(p,n) ≤ r_sq
        r = p .- n
        @inbounds s.storage[f_i] = s.storage[f_i] .+ (1.0, 0.0, r[1], r[2])
        @inbounds s.storage[f_j] = s.storage[f_j] .+ (1.0, -1.0, -r[1], -r[2])
    end
    (s,p,f_i,r_sq, max_frames)
end
    
function _build_normal_equations(frames, cl, max_dist, max_frames = length(frames))
    n = length(frames)
    r_sq = max_dist^2
        
    K_lock = Threads.ReentrantLock()
    K = zeros(n,n)
        
    b_lock = Threads.ReentrantLock()
    b_x = zeros(n)
    b_y = zeros(n)
    K_D = zeros(n)
        
    states = [NE(n) for _ in 1:Threads.nthreads()]
    
    Threads.@threads for ps in frames
        f_i = ps[1][2]
        state = states[Threads.threadid()]
        zero!(state)
        
        for (p, _) in ps
            state = neighbors_fold(_update_ne, (state, p, f_i, r_sq, max_frames), cl, p)[1]
        end
        
            lock(K_lock)  
    @simd   for i in 1:n
        
                @inbounds K_D[i] += state.storage[i][1]
                @inbounds K[i,f_i] += state.storage[i][2]
            end
            unlock(K_lock)
            lock(b_lock) 
                @simd for i in 1:n
                    @inbounds b_x[i] += state.storage[i][3]
                    @inbounds b_y[i] += state.storage[i][4]
                end
        unlock(b_lock) 
    end
    K = K + K' + Diagonal(K_D)
    (Symmetric(K), b_x, K, b_y)
end

end
