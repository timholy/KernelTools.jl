# Macros:
#   reordering loops---done  (add @simd?)
#   tiling
#   parallelization
#   loop unrolling macro?
#   automatic bounds-checking?
#   writing generated code to an executable file

# reordering: @test_looporder and @loophoist (the latter for being after you've settled on a looporder).
# add?: @test_looporder_simd and @loophoist_simd

# tiling API:
#    @tile tilesizes loopnest
#    @tile tilesizes pre loopnest
# e.g.,
#    @tile (m,256,k,16,n,16) (Bt[n,k] = B[k,n]) begin
#         @loophoist for n = 1:N,  k = 1:K, m = 1:M
#             C[m,n] += A[m,k] * Bt[n,k]
#         end
#     end
# and
#    @tile((x,256,y,16), (blur_y[x,y] = A[x,y-1]+A[x,y]+A[x,y+1]) begin
#         for y = interior_y, x = interior_x
#             blur_xy[x,y] = (1.0/9)*(blur_y[x-1,y] + blur_y[x,y] + blur_y[x+1,y])
#         end
#     end
# Need to support:
#    pre-expression: automatic computation of the size of Bt, auto-generation of the necessary loops
#    @test_tile, a syntax that runs & times the computation for many different tile sizes
# Alternative syntax:
#    @tile (m,mo,256,k,ko,16,n,no,16) (Bt[n,k] = B[k,n]) begin
# This should allow you to combine this with
#    @test_looporder @tile (...)
# and
#    @thread (mo, no) @tile (...)
#
# Parallelization:
#    @thread n for n = 1:size(A,2), ...
#    @thread (m,n) for n = 1:size(A,2), ...
# An issue will be allocation of temporaries in tiles; how to express this, if @thread
# is supposed to a separate step? Make need @tile_thread? Or, any allocations that come
# before the for loop could be interpreted as needing to be done once for each thread.

### loop ordering & hoisting
macro test_looporder(ex)
    loopVars, loopRanges = extract_loopvars(ex)
    nvars = length(loopVars)
    body = ex.args[2]
    exout = Expr(:block, Any[])
    for lperm in permutations(1:nvars)
        loopvars = loopVars[lperm]
        loopranges = loopRanges[lperm]
        newbody = gen_hoisted(body, loopvars, loopranges)
        push!(exout.args, :(println($loopvars); @time $newbody))
    end
    exout
end

macro loophoist(ex)
    loopvars, loopranges = extract_loopvars(ex)
    body = ex.args[2]
    gen_hoisted(body, loopvars, loopranges)
end


function extract_loopvars(ex::Expr)
    if ex.head != :for
        error("Must be a for expression")
    end
    loopVars = Symbol[]
    loopRanges = Any[]
    if ex.args[1].head == :(=)
        push!(loopVars, ex.args[1].args[1])
        push!(loopRanges, ex.args[1].args[2])
    elseif ex.args[1].head == :block
        for a in ex.args[1].args
            a.head == :(=) || error("Malformed loop-index block")
            push!(loopVars, a.args[1])
            push!(loopRanges, a.args[2])
        end
    else
        error("Cannot parse loop variables")
    end
    loopVars, loopRanges
end

function gen_hoisted(body, loopvars, loopranges)
    nvars = length(loopvars)
    hoisted, newbody = hoist(body, loopvars)
    # Ordering hoists by loopstage
    hoistex = collect(keys(hoisted))
    vals = collect(values(hoisted))
    hoistsym = [v[1] for v in vals]
    hoiststage = [v[2] for v in vals]
    p = sortperm(hoiststage)
    hoistex = hoistex[p]
    hoistsym = hoistsym[p]
    hoiststage = hoiststage[p]
    ## Code generation
    newbody = esc(newbody)
    k = length(hoiststage)
    for ls = nvars:-1:1
        newbody = quote
            for $(esc(loopvars[ls])) = $(esc(loopranges[ls]))
                $newbody
            end
        end
        while k > 0 && hoiststage[k] == ls-1
            newbody = quote
                $(esc(hoistsym[k])) = $(esc(hoistex[k]))
                $newbody
                $(esc(hoistex[k])) = $(esc(hoistsym[k]))
            end
            k -= 1
        end
    end
    newbody
end

hoist(body::Expr, varnames) = hoist!(Dict{Expr, (Symbol, Int)}(), copy(body), varnames)
function hoist!(hoisted::Dict{Expr, (Symbol, Int)}, exex::Expr, varnames)
    # This focuses on the arguments rather than the head expression, because we need to be able to replace
    # expressions inside the container. One place where this will have trouble is if the top-level expression
    # is a single :ref expression, but that doesn't seem likely/useful in a real computation.
    for k = 1:length(exex.args)
        ex = exex.args[k]
        if isa(ex, Expr)
            # Check inside ex
            hoist!(hoisted, ex, varnames)
            # Now check ex itself
            if ex.head == :ref
                ls = 0  # loopstage for this reference
                for i = 2:length(ex.args)
                    ls = max(ls, loopstage_indexes(ex.args[i], varnames))
                end
                if ls < length(varnames)
                    # We can hoist. But first check whether this expression is already in the list
                    if haskey(hoisted, ex)
                        s = hoisted[ex][1]
                    else
                        s = ktgensym()
                        hoisted[ex] = (s, ls)
                    end
                    exex.args[k] = s
                end
            end
        end
    end
    hoisted, exex
end

loopstage_indexes(s::Symbol, varnames) = indexin_scalar(s, varnames)
function loopstage_indexes(s::Expr, varnames)
    ls = 0
    for a in s.args
        ls = max(ls, loopstage_indexes(a, varnames))
    end
    ls
end
loopstage_indexes(arg, varnames) = 0

# A gensym-variant that generates names that can be used to write kernels to a parsable file
# (The `##32199` symbols produced by gensym() are not parsable)
let counter = 0
global ktgensym
global ktgensym_counter_reset
ktgensym_counter_reset() = (counter = 0)
function ktgensym()
    counter += 1
    symbol(string("_kt_", counter))
end
end
ktgensym(s::Symbol) = symbol("_kt_"*string(s))

# like indexin but first argument must be a scalar
function indexin_scalar(a, b)
    i = 1
    for bb in b
        if a == bb
            return i
        end
        i += 1
    end
    0
end


### Tiling ###

macro tile(tilevariables, args...)
    _tile(tilevariables, args...)
end
function _tile(tilevariables, args...)
    if length(args) == 2
        pre = args[1]
        body = args[2]
    elseif length(args) == 1
        pre = nothing
        body = args[1]
    end
    loopvars = Symbol[]
    tilevars = Symbol[]
    tilesize = Int[]
    k = 1
    (isa(tilevariables, Expr) && tilevariables.head == :tuple) || error("First argument must be a tuple-expression")
    tilevariables = tilevariables.args
    while k < length(tilevariables)
        isa(tilevariables[k], Symbol) || error("error parsing tilevariables")
        push!(loopvars, tilevariables[k])
        k += 1
        if isa(tilevariables[k], Int)
            tv = symbol("_kt_outer_"*string(loopvars[end]))
        elseif isa(tilevariables[k], Symbol)
            tv = tilevariables[k]
            k += 1
        else
            error("error parsing tilevariables")
        end
        push!(tilevars, tv)
        push!(tilesize, tilevariables[k])
        k += 1
    end
    tilesizesym = [symbol("_kt_tsz_"*string(lv)) for lv in loopvars]
    ex = gen_tiled(loopvars, tilevars, tilesizesym, pre, body)
    settilesize = Expr(:block, Any[:($(esc(tilesizesym[i])) = $(tilesize[i])) for i = 1:length(tilesize)]...)
#     settilesize = Expr(:block, Any[:($(tilesizesym[i]) = $(tilesize[i])) for i = 1:length(tilesize)]...)
    return quote
        $settilesize
        $ex
    end
end

# Thoughts: should use more Dicts to make this easier?
function gen_tiled(loopvars, outervars, tilesizesym, pre, body::Expr)
    initblocks = Any[]  # These will allocate temporaries used on a per-tile basis
    preblocks = Any[]   # These run at the beginning of working on each tile
    innervars = [symbol("_kt_inner_"*string(lv)) for lv in loopvars]
    # Look for the initial "for" statement. It might be buried inside @loophoist
    body = copy(body)
    bodymod = body  # points to the head of the "real" expression
    if bodymod.head == :block
        bodymod = Base.is_linenumber(bodymod.args[1]) ? bodymod.args[2] : bodymod.args[1]
    end
    if (bodymod.head == :macrocall && bodymod.args[1] == symbol("@loophoist")) || is_quoted_macrocall(bodymod.args[1], :KernelTools, symbol("@loophoist"))
        bodymod = bodymod.args[2]
    end
    @assert bodymod.head == :for
    looprangeexpr = bodymod.args[1]
    looprangedict = Dict{Symbol,Expr}()
    if looprangeexpr.head == :block
        for ex in looprangeexpr.args
            looprangedict[ex.args[1]] = ex.args[2]
        end
    else
        looprangedict[looprangeexpr.args[1]] = looprangeexpr.args[2]
    end
    # Handle any tilewise temporaries
    if pre != nothing
        ## Infer the sizes of the temporaries
        # Extract all references, and determine which temporaries will be needed
        tmpnames = assignments(pre)  # these are the temporary arrays we'll want
        refs = refexprs(pre, skiplhs=true)
        refexprs!(refs, body)        # refs contains exprs for all :ref expressions, other than setindex! statements in pre
        # Gather all :ref expressions for the same array
        byarray = Dict()
        for r in refs
            arraysym = r.args[1]
            indexes = r.args[2:end]
            if !haskey(byarray, arraysym)
                byarray[arraysym] = Any[indexes]
            else
                push!(byarray[arraysym], indexes)
            end
        end
        # Compute bounds for each temporary
        offsetsyms = Any[]
        sizesyms = Any[]
        boundsblock = Any[]
        for (arrayname, indexes) in byarray
            if !in(arrayname, tmpnames)
                continue
            end
            osyms, ssyms = constructbounds!(boundsblock, arrayname, indexes, loopvars, tilesizesym)
            push!(offsetsyms, osyms)
            push!(sizesyms, ssyms)
        end
        # Annotate the bounds computation so future analysis will know what this is about
        push!(initblocks, Expr(:block, esc(:(KT_BOUNDSCOMP_BLOCK = 1)), boundsblock...))
        # Allocate the temporaries. Now that we have the sizes, the only problem left is the types.
        # We'll compute an element of each temporary and use typeof.
        allocblock = Any[]
        for (i,Asym) in enumerate(tmpnames)
            # Find the first statement in pre that defines each temporary
            ex, _ = assignmentexpr(pre, Asym)
            # Change all indexes on the RHS to 1
            ex1 = index1!(copy(ex.args[2]))
            push!(allocblock, esc(:($Asym = Array(typeof($ex1), $(sizesyms[i]...)))))
        end
        push!(initblocks, Expr(:block, esc(:(KT_ALLOC_BLOCK = 1)), allocblock...))
        # In a tile, initialize each temporary
        for (i,Asym) in enumerate(tmpnames)
            ex, _ = assignmentexpr(pre, Asym)
            osyms = offsetsyms[i]
            ssyms = sizesyms[i]
            lv = ex.args[1].args[2:end]  # the local loopvars used for this temporary
            keep = indexin(loopvars, lv) .> 0
            sum(keep) == length(lv) || error("Pre expressions must use the same indexing variables as the overall loop")
            lv, ov, iv = loopvars[keep], outervars[keep], innervars[keep]
            loopranges = [esc(:($(iv[d]) = 1-$(osyms[d]):min($(ssyms[d])-$(osyms[d]), last($(looprangedict[lv[d]]))-$(ov[d])))) for d = 1:length(iv)]
            push!(preblocks, Expr(:for, Expr(:block, loopranges...), tileindex(esc(ex), lv, ov, iv, tmpnames, offsetsyms)))
        end
        # Replace the body indexing to be tile-specific
        bodymod.args[2] = tileindex(bodymod.args[2], loopvars, outervars, innervars, tmpnames, offsetsyms)
    else
        bodymod.args[2] = tileindex(bodymod.args[2], loopvars, outervars, innervars, Symbol[], [])
    end
    # Create the outer loop nest and modify the inner loop nest to just iterate over a tile
    loopranges = Expr[]
    if length(loopvars) > 1
        looprangeexprs = bodymod.args[1].args  # it's inside a :block statement
        for (i,lv) in enumerate(loopvars)
            ind = find(ex->ex.args[1] == lv, looprangeexprs)
            @assert length(ind) == 1
            j = ind[1]
            rng = looprangeexprs[j].args[2]
            push!(loopranges, tilerange(outervars[i], tilesizesym[i], rng))
            looprangeexprs[j] = :($(innervars[i]) = 0:min($(tilesizesym[i])-1,last($rng)-$(outervars[i])))
        end
    else
        push!(loopranges, tilerange(outervars[1], tilesizesym[1], bodymod[1].args[1]))
        bodymod.args[1] = esc(:($(innervars[1]) = 0:$(tilesizesym[1])-1))
    end
    # Create the final expression
    block = Any[]
    if !isempty(initblocks)
        push!(block, initblocks...)
    end
    body = esc(body)
    if !isempty(preblocks)
        body = Expr(:block, Expr(:block, esc(:(KT_TILEWISE_BLOCK = 1)), preblocks...), body)
    end
    push!(block, Expr(:for, Expr(:block, loopranges...), body))
    Expr(:block, block...)
end

# On output, stmnts will have the expressions needed to compute the size parameters (offset, size) of the given array
function constructbounds!(stmnts::Vector{Any}, arrayname, indexes, loopvars, tilesizesym)
    nd = length(indexes[1])
    if !all(x->length(x) == nd, indexes)
        error("indexes are not all of the same dimensionality: $indexes")
    end
    # Parse all indexing operations on the temporary
    indexexprs = [Any[] for d = 1:nd]
    for I in indexes
        for d = 1:nd
            ind = AffineIndex(I[d])
            push!(indexexprs[d], ind.offset)   # when the inner tile variable is 0
            varindex = indexin_scalar(ind.sym, loopvars)
            if varindex != 0
                # when the inner tile variable is tilesizesym-1
                push!(indexexprs[d], :($(ind.coeff) * ($(tilesizesym[varindex])-1) + $(ind.offset)))
            end
        end
    end
    # Compute offset and size expressions
    offsetsyms = Array(Symbol, nd)
    sizesyms = Array(Symbol, nd)
    for d = 1:nd
        offsetsym = symbol("_kt_offset_"*string(arrayname)*"_"*string(d))
        expr = Expr(:call, :min, indexexprs[d]...)
        push!(stmnts, esc(:($offsetsym = 1 - $expr)))
        sizesym = symbol("_kt_size_"*string(arrayname)*"_"*string(d))
        expr = copy(expr)
        expr.args[1] = :max
        push!(stmnts, esc(:($sizesym = $offsetsym + $expr)))
        offsetsyms[d] = offsetsym
        sizesyms[d] = sizesym
    end
    offsetsyms, sizesyms
end

# Get the symbol associated with assignment statements, e.g., in :(A[5] = 7) return :A
assignments(ex::Expr) = (asgn = Symbol[]; assignments!(asgn, ex))

function assignments!(asgn, ex::Expr)
    if in(ex.head, (:(=), :(+=), :(-=), :(*=), :(/=)))
        if isa(ex.args[1], Expr)
            a = ex.args[1]::Expr
            if a.head == :ref
                push!(asgn, a.args[1]::Symbol)
            end
        end
    else
        for i = 1:length(ex.args)
            assignments!(asgn, ex.args[i])
        end
    end
    asgn
end
assignments!(asgn, a) = asgn

# Get the expression that first assigns a particular array symbol
function assignmentexpr(ex::Expr, Asym::Symbol)
    if in(ex.head, (:(=), :(+=), :(-=), :(*=), :(/=)))
        if isa(ex.args[1], Expr)
            a = ex.args[1]::Expr
            if a.head == :ref && a.args[1] == Asym
                return ex, true
            end
        end
    end
    for i = 1:length(ex.args)
        if isa(ex.args[i], Expr)
            newex, found = assignmentexpr(ex.args[i]::Expr, Asym)
            if found
                return newex, true
            end
        end
    end
    return ex, false
end

# Get all reference expressions, i.e., :(A[i,j]). Optionally skip those that occur on the
# lhs of an assignment.
refexprs(ex::Expr; skiplhs=false) = (refs = Expr[]; refexprs!(refs, ex, skiplhs=skiplhs))

function refexprs!(refs, ex::Expr; skiplhs=false)
    if in(ex.head, (:(=), :(+=), :(-=), :(*=), :(/=)))
        if isa(ex.args[1], Expr)
            a = ex.args[1]::Expr
            if !skiplhs && a.head == :ref
                push!(refs, a)
            end
        end
        refexprs!(refs, ex.args[2], skiplhs=skiplhs)
    else
        if ex.head == :ref
            push!(refs, ex)
        end
        for i = 1:length(ex.args)
            refexprs!(refs, ex.args[i], skiplhs=skiplhs)
        end
    end
    refs
end
refexprs!(refs, s; skiplhs=false) = refs

# Change all indexes to 1.
# This is used for evaluation in the service of type inference
function index1!(ex::Expr)
    if ex.head == :ref
        for i = 2:length(ex.args)
            ex.args[i] = 1
        end
    else
        for i = 1:length(ex.args)
            if isa(ex.args, Expr)
                index1!(ex.args[i]::Expr)
            end
        end
    end
    ex
end

# Convert untiled indexing statements to tiled indexing statements
# For arrays that cover the whole domain, i -> i_outer + i_inner
# For temporary arrays defined just on a tile, i -> i_inner + array_i_offset
# Note that the pattern-matching is based on loopvars and tmpnames, and everything
# else is based on the position (i.e., index) within outervars/innervars/offsetsyms.
function tileindex(ex::Expr, loopvars, outervars, innervars, tmpnames, offsetsyms)
    domainexprs = [:($(outervars[i]) + $(innervars[i]))  for i = 1:length(loopvars)]
    exout = tileindex!(copy(ex), loopvars, domainexprs, innervars, tmpnames, offsetsyms)
    exout
end

function tileindex!(ex::Expr, loopvars, domainexprs, innervars, tmpnames, offsetsyms)
    if ex.head == :ref
        sym = ex.args[1]
        ind = indexin_scalar(sym, tmpnames)
        if ind > 0
            osyms = offsetsyms[ind]
            for i = 2:length(ex.args)
                ex.args[i] = replaceindex!(ex.args[i], loopvars, innervars, osyms[i-1])
            end
        else
            for i = 2:length(ex.args)
                ex.args[i] = replaceindex!(ex.args[i], loopvars, domainexprs)
            end
        end
    end
    for i = 1:length(ex.args)
        if isa(ex.args[i], Expr)
            tileindex!(ex.args[i]::Expr, loopvars, domainexprs, innervars, tmpnames, offsetsyms)
        elseif isa(ex.args[i], Symbol)
            s = ex.args[i]::Symbol
            ind = indexin_scalar(s, loopvars)
            ex.args[i] = ind > 0 ? domainexprs[ind] : s
        end
    end
    ex
end

function replaceindex!(s::Symbol, loopvars, replacevars)
    ind = indexin_scalar(s, loopvars)
    if ind > 0
        return replacevars[ind]
    end
    s
end
function replaceindex!(ex::Expr, loopvars, replacevars)
    for i = 1:length(ex.args)
        ex.args[i] = replaceindex!(ex.args[i], loopvars, replacevars)
    end
    ex
end
function replaceindex!(s::Symbol, loopvars, replacevars, offset::Symbol)
    ind = indexin_scalar(s, loopvars)
    if ind > 0
        return :($(replacevars[ind]) + $offset)
    end
    :($s + $offset)
end
function replaceindex!(ex::Expr, loopvars, replacevars, offset::Symbol)
    for i = 1:length(ex.args)
        ex.args[i] = replaceindex!(ex.args[i], loopvars, replacevars, offset)
    end
    ex
end

tilerange(outersym, tilesizesym, rng::Symbol) = esc(:($outersym = first($rng):$tilesizesym:last($rng)))
function tilerange(outersym, tilesizesym, rng::Expr)
    rng.head == :(:) || error("Must be a range expression")
    length(rng.args) == 2 || error("Sorry, step size not yet supported")
    esc(:($outersym = first($rng):$tilesizesym:last($rng)))
end

is_quoted_macrocall(a, modulename, sym::Symbol) = false
function is_quoted_macrocall(ex::Expr, modulename, sym::Symbol)
    ex.head == :. && ex.args[1] == modulename &&
        ((isa(ex.args[2], QuoteNode) && (ex.args[2]::QuoteNode).value == sym) ||
         (isa(ex.args[2], Expr) && (ex.args[2]::Expr).head == :quote && (ex.args[2]::Expr).args[1] == sym))
end


# Constant-folding for parsing indexes
# Parse indexing expressions like A[2k+1] to extract the coefficient (2), symbol (:k), and offset (1)
# Used for bounds-inference on temporary arrays. Each index must involve at most one symbol.
immutable AffineIndex
    sym::Symbol
    coeff::Int
    offset::Int
end
AffineIndex() = AffineIndex(:nothing, 0, 0)
AffineIndex(s::Symbol) = AffineIndex(s, 1, 0)
AffineIndex(i::Int) = AffineIndex(:nothing, 0, i)
function AffineIndex(ex::Expr)
    if ex.head == :call
        f = ex.args[1]
        if f == :+
            return +(AffineIndex(ex.args[2]), ex.args[3:end]...)
        elseif f == :-
            return -(AffineIndex(ex.args[2]), ex.args[3:end]...)
        elseif f == :*
            return *(AffineIndex(ex.args[2]), ex.args[3:end]...)
        else
            error("Operation $f not recognized")
        end
    else
        error("Expression appears not to be algebraic")
    end
end

(+)(a::AffineIndex, b::Int) = AffineIndex(a.sym, a.coeff, a.offset+b)
(-)(a::AffineIndex, b::Int) = AffineIndex(a.sym, a.coeff, a.offset-b)
(*)(a::AffineIndex, b::Int) = AffineIndex(a.sym, b*a.coeff, a.offset)
(+)(a::AffineIndex, b::Symbol) = a.sym == :nothing ? AffineIndex(b, a.coeff+1, a.offset) : error("cannot evaluate $a + $b")
(-)(a::AffineIndex, b::Symbol) = a.sym == :nothing ? AffineIndex(b, a.coeff-1, a.offset) : error("cannot evaluate $a - $b")
(*)(a::AffineIndex, b::Symbol) = a.sym == :nothing ? AffineIndex(b, a.offset, 0) : error("cannot evaluate $a * $b")
function (+)(a::AffineIndex, b::AffineIndex)
    sym = (a.sym == b.sym) ? a.sym : (a.sym == :nothing ? b.sym : (b.sym == :nothing ? a.sym : error("cannot combine affine expressions with different symbols")))
    AffineIndex(sym, a.coeff+b.coeff, a.offset+b.offset)
end
function (-)(a::AffineIndex, b::AffineIndex)
    sym = (a.sym == b.sym) ? a.sym : (a.sym == :nothing ? b.sym : (b.sym == :nothing ? a.sym : error("cannot combine affine expressions with different symbols")))
    AffineIndex(sym, a.coeff-b.coeff, a.offset-b.offset)
end
function (*)(a::AffineIndex, b::AffineIndex)
    sym = a.sym == :nothing ? b.sym : (b.sym == :nothing ? a.sym : error("cannot combine affine expressions with different symbols"))
    AffineIndex(sym, a.coeff*b.offset + a.offset*b.coeff, a.offset*b.offset)
end
(+)(a::AffineIndex, b::Expr) = a + AffineIndex(b)
(-)(a::AffineIndex, b::Expr) = a - AffineIndex(b)
(*)(a::AffineIndex, b::Expr) = a * AffineIndex(b)
(+)(a::AffineIndex, b...) = a + (+)(AffineIndex(b[1]), b[2:end]...)
(*)(a::AffineIndex, b...) = a * (*)(AffineIndex(b[1]), b[2:end]...)
