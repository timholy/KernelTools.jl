# Macros:
#   reordering loops---done  (add @simd?)
#   tiling---done
#   parallelization
#   loop unrolling macro?
#   automatic bounds-checking of inputs? the infrastructure is basically in place already
#   writing generated code to an executable file

# add?: @test_looporder_simd and @loophoist_simd
#
# Parallelization:
#    @thread n for n = 1:size(A,2), ...
#    @thread (m,n) for n = 1:size(A,2), ...
# An issue will be allocation of temporaries in tiles; how to express this, if @thread
# is supposed to a separate step? Make need @tile_thread? Or, any allocations that come
# before the for loop could be interpreted as needing to be done once for each thread.

const assignment_heads = (:(=), :(+=), :(-=), :(*=), :(/=))

# Debugging: show useful information. Recognized settings:
#    :inferredbounds    Shows the sizes and offsets of temporaries
#    :tileindex         Shows the current outer tile variable values
#    :bodyindex         Shows (outer, inner) iteration variables in the body
#    :preindex          Shows (outer, inner) iteration variables in the pre-expression(s)
const KT_DEBUG = Dict{Symbol, Any}()
function setdebug(sym::Symbol, val)
    global KT_DEBUG
    KT_DEBUG[sym] = val
end

### loop ordering & hoisting
# In a loop nest like this:
# for n = 1:N
#     for m = 1:M
#         for k = 1:K
#             C[k, m, n] = A[m,n]*B[k]
#         end
#     end
# end
# the expression A[m, n] might be replaced by a stack-allocated variable `sym` assigned
# prior to the k loop.
type HoistInfo
    sym::Symbol
    loopstage::Int    # how many nestings deep at which the indexes are constant
    isassignedto::Bool
end

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
    hoistsym = [v.sym for v in vals]
    hoiststage = [v.loopstage for v in vals]
    hoistassignedto = [v.isassignedto for v in vals]
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
            reassign = hoistassignedto[k] ? (:($(esc(hoistex[k])) = $(esc(hoistsym[k])))) : :nothing
            newbody = quote
                $(esc(hoistsym[k])) = $(esc(hoistex[k]))
                $newbody
                $reassign
            end
            k -= 1
        end
    end
    newbody
end

hoist(body::Expr, varnames) = hoist!(Dict{Expr, HoistInfo}(), copy(body), varnames)
function hoist!(hoisted::Dict{Expr, HoistInfo}, exex::Expr, varnames)
    # This examines the arguments rather than the head expression, because we need to be able to replace
    # expressions inside the "container." One place where this will have trouble is if the top-level expression
    # is a single :ref expression, but that doesn't seem likely/useful in a real computation.
    isassignment = in(exex.head, assignment_heads)
    for k = 1:length(exex.args)
        isassignedto = isassignment && k == 1
        if isa(exex.args[k], Expr)
            ex::Expr = exex.args[k]
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
                        hinfo = hoisted[ex]
                        s = hinfo.sym
                        if isassignedto
                            hinfo.isassignedto = isassignedto
                        end
                    else
                        s = ktgensym()
                        hoisted[ex] = HoistInfo(s, ls, isassignedto)
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

macro tile(tilesettings, args...)
    _tile(tilesettings, args...)
end
function _tile(tilesettings, args...)
    if length(args) == 2
        pre = args[1]
        body = args[2]
    elseif length(args) == 1
        pre = nothing
        body = args[1]
    end
    tilevars = Symbol[]  # these are the variables we will be tiling over. Might be a subset of all loopvars.
    outervars = Symbol[] # the names used for the outer variables (the tile indexes)
    tilesize = Int[]     # the tile size (i.e., the step size for each outer variable)
    k = 1
    (isa(tilesettings, Expr) && tilesettings.head == :tuple) || error("First argument must be a tuple-expression")
    while k < length(tilesettings.args)
        isa(tilesettings.args[k], Symbol) || error("error parsing tilesettings")
        push!(tilevars, tilesettings.args[k])
        k += 1
        if isa(tilesettings.args[k], Int)
            tv = symbol("_kt_outer_"*string(tilevars[end]))
        elseif isa(tilesettings.args[k], Symbol)
            tv = tilesettings.args[k]
            k += 1
        else
            error("error parsing tilesettings")
        end
        push!(outervars, tv)
        push!(tilesize, tilesettings.args[k])
        k += 1
    end
    # Create variables to hold the value of the tilesize.
    # We could have hard-coded these sizes, but using a variable makes it easier to
    # write code to loop over different settings (as in @test_tile).
    tilesizesym = [symbol("_kt_tsz_"*string(tv)) for tv in tilevars]
    ex = gen_tiled(tilevars, outervars, tilesizesym, pre, body)
    # Set the tilesize variables to those specified in the tuple provided by the user.
    settilesize = Expr(:block, Any[:(const $(esc(tilesizesym[i])) = $(tilesize[i])) for i = 1:length(tilesize)]...)
    return quote
        $settilesize
        $ex
    end
end

function gen_tiled(tilevars, outervars, tilesizesym, pre, body::Expr)
    initblocks = Any[]  # Statements to allocate temporaries used on a per-tile basis
    preblocks = Any[]   # Statements that run at the beginning of working on each tile
    innervars = [symbol("_kt_inner_"*string(tv)) for tv in tilevars]
    # Look for the initial "for" statement. It might be buried inside @loophoist
    body = copy(body)
    bodymod = body  # will point to the head of the "real" body expression
    if bodymod.head == :block
        bodymod = Base.is_linenumber(bodymod.args[1]) ? bodymod.args[2] : bodymod.args[1]
    end
    if is_macrocall(bodymod, :KernelTools, symbol("@loophoist"))
        bodymod = bodymod.args[2]
    end
    if bodymod.head != :for
        error("@tile requires that the body expression starts with a for loop; got ", bodymod)
    end
    # Extract all loop variables and their ranges (even those that we're not tiling over)
    looprangeexpr = bodymod.args[1]
    looprangedict = Dict{Symbol,Any}()    # might be an expression, :(1:5), or a symbol, :inner_x
    if looprangeexpr.head == :block
        # A multidimensional loop
        for ex in looprangeexpr.args
            looprangedict[ex.args[1]::Symbol] = copy(ex.args[2])
        end
    else
        # A single-variable loop
        looprangedict[looprangeexpr.args[1]::Symbol] = copy(looprangeexpr.args[2])
    end
    # Check that the tiled variables are a subset of the loop variables
    for tv in tilevars
        if !haskey(looprangedict, tv)
            error("Tile variables must be loop variables; did not find ", tv)
        end
    end

    # Create the outer loop nest and modify the inner loop nest to iterate over a tile
    outerlooprangeexprs = Expr[]
    innerlooprangeexprs = bodymod.args[1].head == :block ? bodymod.args[1].args : [bodymod.args[1]]
    outerorder = Symbol[]
    for i = 1:length(innerlooprangeexprs)
        ex = innerlooprangeexprs[i]
        lv = ex.args[1]
        ind = indexin_scalar(lv, tilevars)
        if ind > 0
            rng = looprangedict[lv]
            push!(outerlooprangeexprs, tilerange(outervars[ind], tilesizesym[ind], rng))
            push!(outerorder, lv)
            innerlooprangeexprs[i] = :($(innervars[ind]) = 0:min($(tilesizesym[ind])-1,last($rng)-$(outervars[ind])))
        end
    end
    if bodymod.args[1].head != :block
        bodymod.args[1] = innerlooprangeexprs[1]
    end
    p = sortperm(indexin(outerorder, tilevars))  # put in order specified by user
    outerlooprangeexprs = outerlooprangeexprs[p]

    # Handle any tilewise temporaries
    if pre != nothing
        ## Infer the sizes of the temporaries
        # Extract all references, and determine which temporaries will be needed
        tmpnames = assignments(pre)  # these are the temporary arrays we'll need
        refs = refexprs(pre, skiplhs=true)
        refexprs!(refs, body)        # refs contains exprs for all :ref expressions, other than setindex! statements in pre
        # Gather all indexes used for a given array
        # If A is used as A[x,y] and A[x+1,y], then
        #   indexes_by_array[:A] == Vector{Any}[[:x,:y], [:(x+1),:y]]
        indexes_by_array = Dict{Symbol,Vector{Vector{Any}}}()
        for r in refs
            arraysym = r.args[1]
            indexes = r.args[2:end]
            if !haskey(indexes_by_array, arraysym)
                indexes_by_array[arraysym] = Vector{Any}[]
            end
            push!(indexes_by_array[arraysym], indexes)
        end
        # From the set of indexing operations, compute bounds for each temporary.
        # This includes what are usually called "ghost cells" beyond the edges of tiles, so
        # that expressions like (tmparray[x-1] + tmparray[x+1]) will be in-bounds.
        # Temporaries will be indexed with x = innervar + offset, where innervar starts at 0.
        # The offset compensates for the smallest index in the stencil.
        offsetsyms = Any[]    # Names of variables holding the offsets for each temporary
        sizesyms = Any[]      # Names of variables holding the sizes of each temporary
        lastsyms = Any[]      # Names of variables holding the largest value of (outer+inner) along each axis
                              # at which the temporary needs to be initialized.
                              # (if tmp[x] = A[x] and out[x] = tmp[x]+tmp[x+1], we need to evaluate
                              #  tmp at one higher x than the range used for out)
        boundsblock = Any[]   # Statements needed to compute the bounds
        for (arrayname, indexes) in indexes_by_array
            if !in(arrayname, tmpnames)
                continue
            end
            osyms, ssyms, lsyms = constructbounds!(boundsblock, arrayname, indexes, tilevars, tilesizesym, looprangedict)
            push!(offsetsyms, osyms)
            push!(sizesyms, ssyms)
            push!(lastsyms, lsyms)
        end
        # Annotate the bounds computation so future analysis will know what this is about
        push!(initblocks, Expr(:block, esc(:(KT_BOUNDSCOMP_BLOCK = 1)), boundsblock...))
        # Allocate the temporaries. Now that we have the sizes, the only problem left is the types.
        # We'll compute one element of each temporary and use typeof.
        allocblock = Any[]
        for (i,Asym) in enumerate(tmpnames)
            # Find the first statement in pre that defines each temporary
            ex, _ = assignmentexpr(pre, Asym)
            # Change all indexes on the RHS to offsets. This is a safe place to evaluate it
            outfirst = [:(first($(looprangedict[tv]))) for tv in tilevars]
            ex1 = tileindex(ex.args[2], tilevars, outfirst, zeros(Int, length(tilevars)), tmpnames, offsetsyms[i])
            push!(allocblock, esc(:($Asym = Array(typeof($ex1), $(sizesyms[i]...)))))
        end
        push!(initblocks, Expr(:block, esc(:(KT_ALLOC_BLOCK = 1)), allocblock...))
        # Now we build code that will run inside the tile loops
        # Initialize each temporary
        for (i,Asym) in enumerate(tmpnames)
            # Here's a design problem: what about temporaries that take more than one line to compute?
            ex = assignmentblock(pre, Asym)  # find the expression for calculating the chosen temporary
            exa, _ = assignmentexpr(ex, Asym)
            osyms = offsetsyms[i]
            ssyms = sizesyms[i]
            lsyms = lastsyms[i]
            tv = exa.args[1].args[2:end]  # the local tilevars used for initializing this temporary
            ind = indexin(tv, tilevars)
            any(ind.==0) && error("Pre expressions must use the same indexing variables as the overall loop")
            ov, iv = outervars[ind], innervars[ind]
            loopranges = [esc(:($(iv[d]) = 1-$(osyms[d]):min($(ssyms[d])-$(osyms[d]), $(lsyms[d])-$(ov[d])))) for d = 1:length(iv)]
            indswap = indexin(outerorder, tv)
            indswap = indswap[indswap .> 0]
            loopranges = loopranges[indswap]
            initbody = tileindex(esc(ex), tv, ov, iv, tmpnames, offsetsyms)
            if get(KT_DEBUG, :preindex, false)
                initbody = Expr(:block, esc(Expr(:macrocall, symbol("@show"), Expr(:tuple, iv...))), initbody)
                push!(preblocks, :(println("Initializing ", $(Expr(:quote, Asym)))))
            end
            push!(preblocks, Expr(:for, Expr(:block, loopranges...), initbody))
        end
        # Replace the body indexing to be tile-specific
        bodymod.args[2] = tileindex(bodymod.args[2], tilevars, outervars, innervars, tmpnames, offsetsyms)
    else
        bodymod.args[2] = tileindex(bodymod.args[2], tilevars, outervars, innervars, Symbol[], [])
    end
    # Create the final expression
    block = Any[]
    if !isempty(initblocks)
        push!(block, initblocks...)
    end
    if get(KT_DEBUG, :bodyindex, false)
        unshift!(bodymod.args[2].args, Expr(:macrocall, symbol("@show"), Expr(:tuple, innervars...)))
    end
    body = esc(body)
    if get(KT_DEBUG, :tileindex, false)
        unshift!(preblocks, esc(Expr(:macrocall, symbol("@show"), Expr(:tuple, outervars...))))
    end
    if !isempty(preblocks)
        body = Expr(:block, Expr(:block, esc(:(KT_TILEWISE_BLOCK = 1)), preblocks...), body)
    end
    push!(block, Expr(:for, Expr(:block, outerlooprangeexprs...), body))
    Expr(:block, block...)
end

# On output, stmnts will have the expressions needed to compute the size parameters (offset, size, last) of the given
# temporary array.
# See more detailed description of the outputs above in the call to constructbounds!
function constructbounds!(stmnts::Vector{Any}, arrayname, indexes, tilevars, tilesizesym, looprangedict)
    nd = length(indexes[1])
    if !all(x->length(x) == nd, indexes)
        error("indexes are not all of the same dimensionality: $indexes")
    end
    # Parse all indexing operations on the temporary
    indexexprs = [Any[] for d = 1:nd]  # Indexing expressions evaluated at first&last indexes of a tile
    lastexprs = [Any[] for d = 1:nd]    # Used for the final tile along an axis
    for I in indexes
        for d = 1:nd
            ind = AffineIndex(I[d])
            tvindex = indexin_scalar(ind.sym, tilevars)
            rng = looprangedict[ind.sym]
            if tvindex != 0
                # when the inner tile variable is at min (0)
                push!(indexexprs[d], ind.offset)
                # when the inner tile variable is at max (tilesizesym-1)
                push!(indexexprs[d], :($(ind.coeff) * ($(tilesizesym[tvindex])-1) + $(ind.offset)))
            else
                # It's not a variable we're tiling over, so use the whole range
                push!(indexexprs[d], :(first($rng)))
                push!(indexexprs[d], :(last($rng)))
            end
            push!(lastexprs[d], :($(ind.coeff) * last($rng) + $(ind.offset)))
        end
    end
    # Add statements that compute the offset and size expressions
    offsetsyms = Array(Symbol, nd)
    sizesyms = Array(Symbol, nd)
    lastsyms = Array(Symbol, nd)
    toshow = get(KT_DEBUG, :inferredbounds, false)
    for d = 1:nd
        tag = string(arrayname)*"_"*string(d)
        # The offset is computed in terms of the minimum value of all indexing operations
        offsetsym = symbol("_kt_offset_"*tag)
        expr = Expr(:call, :min, indexexprs[d]...)
        push!(stmnts, esc(:(const $offsetsym = 1 - $expr)))
        sizesym = symbol("_kt_size_"*tag)
        lastsym = symbol("_kt_last_"*tag)
        # The size is computed in terms of the maximum value of all indexing operations
        expr = copy(expr)
        expr.args[1] = :max
        push!(stmnts, esc(:(const $sizesym = $offsetsym + $expr)))
        # The lastvalue is the maximum of all statements at total loop edges
        expr = length(lastexprs[d]) > 1 ? Expr(:call, :max, lastexprs[d]...) : lastexprs[d][1]
        push!(stmnts, esc(:(const $lastsym = $expr)))
        offsetsyms[d] = offsetsym
        sizesyms[d] = sizesym
        lastsyms[d] = lastsym
        if toshow
            push!(stmnts, esc(:(@show $offsetsym)))
            push!(stmnts, esc(:(@show $sizesym)))
            push!(stmnts, esc(:(@show $lastsym)))
        end
    end
    offsetsyms, sizesyms, lastsyms
end

# Get the symbol associated with assignment statements, e.g., in :(A[5] = 7) return :A
assignments(ex::Expr) = (asgn = Symbol[]; assignments!(asgn, ex))

function assignments!(asgn, ex::Expr)
    if in(ex.head, assignment_heads)
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
function assignmentblock(ex::Expr, Asym::Symbol)
    if ex.head == :block
        args = ex.args
        for a in args
            _, found = assignmentexpr(a)
            if found
                return a
            end
        end
        error("Not found")
    end
    ex
end

function assignmentexpr(ex::Expr, Asym::Symbol)
    if in(ex.head, assignment_heads)
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
    if in(ex.head, assignment_heads)
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

# Convert untiled indexing statements to tiled indexing statements
# For arrays that cover the whole domain, i -> i_outer + i_inner
# For temporary arrays defined just on a tile, i -> i_inner + array_i_offset
# Note that the pattern-matching is based on tilevars and tmpnames, and everything
# else is based on the position (i.e., index) within outervars/innervars/offsetsyms.
function tileindex(ex::Expr, tilevars, outervars, innervars, tmpnames, offsetsyms)
    domainexprs = [:($(outervars[i]) + $(innervars[i]))  for i = 1:length(tilevars)]
    exout = tileindex!(copy(ex), tilevars, domainexprs, innervars, tmpnames, offsetsyms)
    exout
end

function tileindex!(ex::Expr, tilevars, domainexprs, innervars, tmpnames, offsetsyms)
    if ex.head == :ref
        sym = ex.args[1]
        ind = indexin_scalar(sym, tmpnames)
        if ind > 0
            osyms = offsetsyms[ind]
            for i = 2:length(ex.args)
                ex.args[i] = replaceindex!(ex.args[i], tilevars, innervars, osyms[i-1])
            end
        else
            for i = 2:length(ex.args)
                ex.args[i] = replaceindex!(ex.args[i], tilevars, domainexprs)
            end
        end
    end
    for i = 1:length(ex.args)
        if isa(ex.args[i], Expr)
            tileindex!(ex.args[i]::Expr, tilevars, domainexprs, innervars, tmpnames, offsetsyms)
        elseif isa(ex.args[i], Symbol)
            s = ex.args[i]::Symbol
            ind = indexin_scalar(s, tilevars)
            ex.args[i] = ind > 0 ? domainexprs[ind] : s
        end
    end
    ex
end

replaceindex!(arg, tilevars, replacevars) = arg
function replaceindex!(s::Symbol, tilevars, replacevars)
    ind = indexin_scalar(s, tilevars)
    if ind > 0
        return replacevars[ind]
    end
    s
end
function replaceindex!(ex::Expr, tilevars, replacevars)
    for i = 1:length(ex.args)
        ex.args[i] = replaceindex!(ex.args[i], tilevars, replacevars)
    end
    ex
end
replaceindex!(arg, tilevars, replacevars, offset::Symbol) = arg
function replaceindex!(s::Symbol, tilevars, replacevars, offset::Symbol)
    ind = indexin_scalar(s, tilevars)
    if ind > 0
        return :($(replacevars[ind]) + $offset)
    end
#     :($s + $offset)
    s
end
function replaceindex!(ex::Expr, tilevars, replacevars, offset::Symbol)
    for i = 1:length(ex.args)
        ex.args[i] = replaceindex!(ex.args[i], tilevars, replacevars, offset)
    end
    ex
end

tilerange(outersym, tilesizesym, rng::Symbol) = esc(:($outersym = first($rng):$tilesizesym:last($rng)))
function tilerange(outersym, tilesizesym, rng::Expr)
    rng.head == :(:) || error("Must be a range expression")
    length(rng.args) == 2 || error("Sorry, step size not yet supported")
    esc(:($outersym = first($rng):$tilesizesym:last($rng)))
end

is_macrocall(a, modulename, sym::Symbol) = false
function is_macrocall(ex::Expr, modulename, sym::Symbol)
    if ex.head == :macrocall
        ex.args[1] == sym && return true
        if isa(ex.args[1], Expr)
            ex = ex.args[1]::Expr
            return ex.head == :. && ex.args[1] == modulename &&
                ((isa(ex.args[2], QuoteNode) && (ex.args[2]::QuoteNode).value == sym) ||
                 (isa(ex.args[2], Expr) && (ex.args[2]::Expr).head == :quote && (ex.args[2]::Expr).args[1] == sym))
        end
    end
    false
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
