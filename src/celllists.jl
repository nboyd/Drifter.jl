const P2 = NTuple{2, Float64}

@inline _bin_idx(x :: Float64, bin_width :: Float64) = ceil(Int64, x/bin_width)

@inline function _squared_dist(p :: P2, n :: P2)
	(p[1] - n[1])^2 + (p[2] - n[2])^2
end

@inline function _squared_dist(p :: P2,n :: Tuple{P2, T}) where{T}
	(p[1] - n[1][1])^2 + (p[2] - n[1][2])^2
end

struct CellList{T}
    cells :: Dict{NTuple{2, Int64}, Vector{Tuple{P2,T}}}
    radius :: Float64
    sentinel :: T
    function CellList{T}(radius,s :: T) where {T}
        new{T}(Dict(), radius, s)
    end
end

function add_point!(t :: CellList{T}, p :: Tuple{P2,T}) where {T}
    k = _bin_idx(p[1][1], t.radius), _bin_idx(p[1][2], t.radius)

    list = if !haskey(t.cells, k)
        t.cells[k] = Tuple{P2,T}[]
    else
        t.cells[k]
    end

    push!(list, p)
end

function CellList(points :: Vector{Tuple{P2, T}}, radius :: Float64, s:: T) where {T}
    t = CellList{T}(radius, s)
    for p in points
        add_point!(t, p)
    end
    t
end

function neighbors_fold(op, state, t :: CellList, p :: P2)
    r_sq = t.radius*t.radius
    offsets = (0,-1,1)

    b_x = _bin_idx(p[1], t.radius)
    b_y = _bin_idx(p[2], t.radius)

    for o_x in offsets, o_y in offsets
        k = (b_x + o_x, b_y + o_y)
        if haskey(t.cells, k)
            for n in t.cells[k]
                #if _squared_dist(p,n) <= r_sq
                    state = op(state, n)
                #end
            end
        end
    end
    state
end

struct Nearest
    p :: P2
end

@inline function (center :: Nearest)((flag, sq_dist, old_nearest), n)
    sd = _squared_dist(center.p, n)
    ifelse(sd < sq_dist, (true, sd, n), (flag, sq_dist, old_nearest))
end

function neighbor(t :: CellList, p :: P2)
    neighbors_fold(Nearest(p),(false, t.radius^2, ((Inf, Inf), t.sentinel)), t, p )
end