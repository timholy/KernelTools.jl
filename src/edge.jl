import Base: start, next, done

immutable Edge
    interior::Bool
    inner::UnitRange{Int}
    outer::UnitRange{Int}
end

start(e::Edge) = (k = first(e.outer); in(k, e.inner) ? last(e.inner)+1 : k)
next(e::Edge, i) = (i, e.interior && in(i+1, e.inner) ? last(e.inner)+1 : i+1)
done(e::Edge, i) = i == last(e.outer)+1

#=
@nedge prototype usage:
for N = 1:3
    @eval begin
        function foo{T}(A::AbstractArray{T,$N}, region)
            interior = [in(d, region) ? 2:size(A,d)-1 : 1:size(A,d) for d = 1:$N]
            @nexprs $N d->(interior_d = interior[d])
            @nloops $N i d->interior[d] begin
                # do something, not worrying about the boundaries
            end
            @nedge $N i d->1:size(A,d) d->interior_d begin
                # do the same thing, this time worrying about the boundaries
            end
            retval
        end
    end
end

=#


macro nedge(N, itersym, outerexpr, innerexpr, args...)
    _nedge(N, itersym, outerexpr, innerexpr, args...)
end

_nedge(N::Int, itersym::Symbol, outersym::Symbol, innerexpr::Expr, args::Expr...) = _nedge(N, itersym, :(d->1:size($outersym,d)), innerexpr, args...)

function _nedge(N::Int, itersym::Symbol, outerexpr::Expr, innerexpr::Expr, args::Expr...)
    outerexpr.head == :-> || error("outerexpr must be an anonymous-function expression")
    innerexpr.head == :-> || error("innerexpr must be an anonymous-function expression")
    1 <= length(args) <= 3 || error("Wrong number of arguments")
    body = args[end]
    ex = Expr(:escape, body)
    firstsym = gensym()           # Boolean scalar; initialization variable
    fastest_edge_sym = gensym()   # Boolean ntuple; true only for the fastest interior dimension
    interior_sym = gensym()       # Boolean ntuple; true if all slower indexes are in the interior
    fastestexpr = Array(Expr, N)  # used during initialization
    for dim = 1:N
        itervar = Cartesian.inlineanonymous(itersym, dim)
        outer = Cartesian.inlineanonymous(outerexpr, dim)
        inner = Cartesian.inlineanonymous(innerexpr, dim)
        interior_cur  = Cartesian.inlineanonymous(interior_sym, dim-1)
        interior_prev = Cartesian.inlineanonymous(interior_sym, dim)
        fastvar = Cartesian.inlineanonymous(fastest_edge_sym, dim)
        # Set up initialization of fastest_edge_sym
        fastestexpr[dim] = Expr(:escape, :($fastvar = $firstsym & ($outer != $inner); $firstsym &= !$fastvar))
        # Set up updating of interior_sym
        interiorexpr = dim > 1 ? (:($interior_cur = $interior_prev & in($itervar, $inner))) : (:(nothing))
        preexpr = length(args) > 1 ? Cartesian.inlineanonymous(args[1], dim) : (:(nothing))
        postexpr = length(args) > 2 ? Cartesian.inlineanonymous(args[2], dim) : (:(nothing))
        ex = quote
            for $(esc(itervar)) in KernelTools.Edge($(esc(fastvar)) & $(esc(interior_prev)), $(esc(inner)), $(esc(outer)))
                $(esc(interiorexpr))
                $(esc(preexpr))
                $ex
                $(esc(postexpr))
            end
        end
    end
    lastinterior = Cartesian.inlineanonymous(interior_sym, N)
    # Initialize fastest_edge_sym and set the last interior_sym flag to be true before we do the loops
    quote
        $(esc(firstsym)) = true
        $(fastestexpr...)
        $(esc(lastinterior)) = true
        $ex
    end
end
