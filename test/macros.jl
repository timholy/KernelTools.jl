import KernelTools
using Base.Test

#### Testing the utility functions

# Convenience function for testing equality of expressions
# Strip line numbers, extraneous :block wrappers, and :escape expressions

# There's one small bit of type-instability: Expr(:block, :a) or Expr(:escape, :a)
# will get turned into a symbol, not an Expr. By confining the type-instability to
# an "outer" wrapper (stripexpr) and making the core algorithm (stripexpr!) type-stable,
# we hope to make this faster. As a consequence, stripexpr! can only modify the args
# of an expression, and can't elide the container expression itself.
function stripexpr(a)
    ex = stripexpr!(copy(a))
    if isa(ex, Expr) && (ex.head == :block || ex.head == :escape) && length(ex.args) == 1
        return ex.args[1]
    end
    ex
end

stripexpr!(a) = a
function stripexpr!(ex::Expr)
    # Strip line number statements
    keep = ![Base.is_linenumber(ex.args[i]) for i = 1:length(ex.args)]
    ex.args = ex.args[keep]
    # Eliminate :escape and unnecessary :block statements
    nargs = length(ex.args)
    for i = 1:nargs
        if isa(ex.args[i], Expr)
            exa = stripexpr!(ex.args[i]::Expr)
            if exa.head == :escape
                length(exa.args) == 1 || error("complex escape expression")
                ex.args[i] = exa.args[1]
            elseif exa.head == :block && length(exa.args) == 1
                ex.args[i] = exa.args[1]
            end
        end
    end
    ex
end

# hoisting expressions that are constant for the purposes of some loops
KernelTools.ktgensym_counter_reset()
ex = KernelTools.gen_hoisted(:(s += A[i]), [:j, :i], [:(1:size(A,2)),:(1:size(A,1))])
@test stripexpr(ex) == stripexpr(:(for j = 1:size(A,2) for i = 1:size(A,1) s += A[i] end end))
ex = KernelTools.gen_hoisted(:(s += A[j]), [:j, :i], [:(1:size(A,2)),:(1:size(A,1))])
@test stripexpr(ex) == stripexpr(:(for j = 1:size(A,2); _kt_1 = A[j]; for i = 1:size(A,1) s += _kt_1 end; nothing; end))

# Symbols associated with setindex!-like statements
@test KernelTools.assignments(:(a = 5)) == Symbol[]
@test KernelTools.assignments(:(a[3] = 5)) == [:a]
@test KernelTools.assignments(:(a[3] += 5)) == [:a]
@test KernelTools.assignments(:(a[3] -= b[3])) == [:a]
@test KernelTools.assignments(:(a *= b[3])) == Symbol[]
@test KernelTools.assignments(:(A[i,j] /= b[3]; c[7] = 8)) == [:A, :c]

# Collecting all reference expressions
# optionally we ignore all those that occur on the left hand side of an assignment
@test KernelTools.refexprs(:(a = 5)) == Expr[]
@test KernelTools.refexprs(:(a[3] = 5)) == [:(a[3])]
@test KernelTools.refexprs(:(a[3] += 5), skiplhs=true) == Expr[]
@test KernelTools.refexprs(:(a[3] -= b[3])) == [:(a[3]), :(b[3])]
@test KernelTools.refexprs(:(a[3] -= b[3]), skiplhs=true) == [:(b[3])]
@test KernelTools.refexprs(:(a *= b[i])) == [:(b[i])]
@test KernelTools.refexprs(:(A[i,j] /= b[3]; c[7] = 8)) == [:(A[i,j]), :(b[3]), :(c[7])]

# Parsing indexes
@test KernelTools.AffineIndex(7) == KernelTools.AffineIndex(:nothing, 0, 7)
@test KernelTools.AffineIndex(:i) == KernelTools.AffineIndex(:i, 1, 0)
@test KernelTools.AffineIndex(:(2k)) == KernelTools.AffineIndex(:k, 2, 0)
@test KernelTools.AffineIndex(:(j+3)) == KernelTools.AffineIndex(:j, 1, 3)
@test KernelTools.AffineIndex(:(3qq-1)) == KernelTools.AffineIndex(:qq, 3, -1)
@test KernelTools.AffineIndex(:(i + 2 - 1)) == KernelTools.AffineIndex(:i, 1, 1)
@test KernelTools.AffineIndex(:(i + 2 + 3i)) == KernelTools.AffineIndex(:i, 4, 2)
@test_throws ErrorException KernelTools.AffineIndex(:(i + 2 + 3j))

## Bounds computations on temporaries for tiling
    # ti, tj = tile sizes for i, j
    # inner indexes ii take values in 0:(ti-1)
    # A statement like A[i] will get turned into A[offset+ii], and A[i-1] into A[offset+ii-1]
    # compute the offset and size of A needed to be able to execute all indexing statements without bounds errors
looprangedict = [:i => :(1:size(A,1)), :j => :(1:size(A,2))]
# A[i]
stmnts = Any[]
offsetsyms, sizesyms, lastsyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:i]], [:i], [:ti], looprangedict)
@test stripexpr(stmnts[1]) == :(const $(offsetsyms[1]) = 1 - min(0, 1*(ti-1)+0))  # ultimately, this will evaluate to 1
@test stripexpr(stmnts[2]) == :(const $(sizesyms[1]) = $(offsetsyms[1]) + max(0, 1*(ti-1)+0))  # ultimately, this will evaluate to ti
# A[i-1]
stmnts = Any[]
offsetsyms, sizesyms, lastsyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:(i-1)]], [:i], [:ti], looprangedict)
@test stripexpr(stmnts[1]) == :(const $(offsetsyms[1]) = 1 - min(-1, 1*(ti-1)+(-1)))  # ultimately, this will evaluate to 2
@test stripexpr(stmnts[2]) == :(const $(sizesyms[1]) = $(offsetsyms[1]) + max(-1, 1*(ti-1)+(-1)))  # ultimately, this will evaluate to ti
# A[i-1] and A[i+1]
stmnts = Any[]
offsetsyms, sizesyms, lastsyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:(i-1)],[:(i+1)]], [:i], [:ti], looprangedict)
@test stripexpr(stmnts[1]) == :(const $(offsetsyms[1]) = 1 - min(-1, 1*(ti-1)+(-1), 1, 1*(ti-1)+1))  # ultimately, this will evaluate to 2
@test stripexpr(stmnts[2]) == :(const $(sizesyms[1]) = $(offsetsyms[1]) + max(-1, 1*(ti-1)+(-1), 1, 1*(ti-1)+1))  # ultimately, this will evaluate to ti+2
# A[i+1,j] and A[i,j+1]
stmnts = Any[]
offsetsyms, sizesyms, lastsyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:(i+1),:j],[:i,:(j+1)]], [:i,:j], [:ti,:tj], looprangedict)
@test stripexpr(stmnts[1]) == :(const $(offsetsyms[1]) = 1 - min(1, 1*(ti-1)+1, 0, 1*(ti-1)+0))
@test stripexpr(stmnts[2]) == :(const $(sizesyms[1]) = $(offsetsyms[1]) + max(1, 1*(ti-1)+1, 0, 1*(ti-1)+0))
@test stripexpr(stmnts[4]) == :(const $(offsetsyms[2]) = 1 - min(0, 1*(tj-1)+0, 1, 1*(tj-1)+1))
@test stripexpr(stmnts[5]) == :(const $(sizesyms[2]) = $(offsetsyms[2]) + max(0, 1*(tj-1)+0, 1, 1*(tj-1)+1))
# Tiling just j for A[i,j]
stmnts = Any[]
offsetsyms, sizesyms, lastsyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:i,:j]], [:j], [:tj], looprangedict)
@test stripexpr(stmnts[3]) == :(const $(lastsyms[1]) = 1*last($(looprangedict[:i])) + 0)
@test stripexpr(stmnts[6]) == :(const $(lastsyms[2]) = 1*last($(looprangedict[:j])) + 0)
# Tiling just j for A[i+1,j]
stmnts = Any[]
offsetsyms, sizesyms, lastsyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:(i+1),:j]], [:j], [:tj], looprangedict)
@test stripexpr(stmnts[3]) == :(const $(lastsyms[1]) = 1*last($(looprangedict[:i])) + 1)
@test stripexpr(stmnts[6]) == :(const $(lastsyms[2]) = 1*last($(looprangedict[:j])) + 0)


##### Loop ordering in @tile
getlooporder(ex::Expr) = getlooporder!(Array(Vector{Symbol}, 0), ex)
function getlooporder!(order, ex::Expr)
    if ex.head == :for
        loopvars = ex.args[1]
        if isa(loopvars, Expr) && (loopvars::Expr).head == :block
            newvars = Symbol[]
            for exlv in (loopvars::Expr).args
                push!(newvars, getloopvar(exlv))
            end
            push!(order, newvars)
        else
            push!(order, [getloopvar(loopvars)])
        end
        getlooporder!(order, ex.args[2])
    else
        for i = 1:length(ex.args)
            if isa(ex.args[i], Expr)
                getlooporder!(order, ex.args[i])
            end
        end
    end
    order
end
function getloopvar(ex::Expr)
    ex.head == :(=) || error("Not an assignment expression")
    ex.args[1]
end

order = getlooporder(macroexpand(quote
        KernelTools.@tile (i,4) for i = 1:10 end
    end))
@test order == Vector{Symbol}[[:_kt_outer_i], [:_kt_inner_i]]
order = getlooporder(macroexpand(quote
        for j = 1:5
            KernelTools.@tile (i,4) for i = 1:10 end
        end
    end))
@test order == Vector{Symbol}[[:j],[:_kt_outer_i], [:_kt_inner_i]]
order = getlooporder(macroexpand(quote
            KernelTools.@tile (i,4) for j = 1:5, i = 1:10 end
    end))
@test order == Vector{Symbol}[[:_kt_outer_i], [:j, :_kt_inner_i]]
order = getlooporder(macroexpand(quote
            KernelTools.@tile (j,2,i,4) for j = 1:5, i = 1:10 end
    end))
@test order == Vector{Symbol}[[:_kt_outer_j,:_kt_outer_i], [:_kt_inner_j, :_kt_inner_i]]
order = getlooporder(macroexpand(quote
            KernelTools.@tile (i,4,j,2) for j = 1:5, i = 1:10 end
    end))
@test order == Vector{Symbol}[[:_kt_outer_i,:_kt_outer_j], [:_kt_inner_j, :_kt_inner_i]]

##### Functional tests of the macros

# Reductions & hoising
A = rand(3,5)
A1 = sum(A, 2)
A2 = sum(A, 1)
A1c = similar(A1); fill!(A1c, 0)
A2c = similar(A2); fill!(A2c, 0)
KernelTools.@loophoist for j = 1:size(A,2), i = 1:size(A,1)
    A1c[i,1] += A[i,j]
end
KernelTools.@loophoist for j = 1:size(A,2), i = 1:size(A,1)
    A2c[1,j] += A[i,j]
end
@test A1c == A1
@test A2c == A2

# Tiling
A = zeros(Int, 5, 6)
KernelTools.@tile (i,4,j,4) begin
    for j = 1:size(A,2), i = 1:size(A,1)
        A[i,j] = i + (j-1)*size(A,1)
    end
end
@test A == reshape(1:length(A), size(A))

# Tiling over a subset of variables
A = zeros(Int, 5, 6)
KernelTools.@tile (j,4) begin
    for j = 1:size(A,2), i = 1:size(A,1)
        A[i,j] = i + (j-1)*size(A,1)
    end
end
@test A == reshape(1:length(A), size(A))

## Tiling matrix multiplication
M = 8
K = 9
N = 7
A = rand(M,K)
B = rand(K,N)
C = A*B
Cc = similar(C)
# Tiling with just a body expression
fill!(Cc, 0)
KernelTools.@tile (m,4,k,4,n,4) begin
    for n = 1:N, k = 1:K, m = 1:M
        Cc[m,n] += A[m,k] * B[k,n]
    end
end
@test_approx_eq C Cc
fill!(Cc, 0)
KernelTools.@tile (m,4,k,4,n,4) begin
    KernelTools.@loophoist for n = 1:N,  k = 1:K, m = 1:M
        Cc[m,n] += A[m,k] * B[k,n]
    end
end
@test_approx_eq C Cc
# Tiling with a pre-expression
fill!(Cc, 0)
KernelTools.@tile (m,4,k,4,n,4) (Bt[n,k] = B[k,n]) begin
    for n = 1:N,  k = 1:K, m = 1:M
        Cc[m,n] += A[m,k] * Bt[n,k]
    end
end
@test_approx_eq C Cc
fill!(Cc, 0)
KernelTools.@tile (m,4,k,4,n,4) (Bt[n,k] = B[k,n]) begin
    KernelTools.@loophoist for n = 1:N,  k = 1:K, m = 1:M
        Cc[m,n] += A[m,k] * Bt[n,k]
    end
end
@test_approx_eq C Cc
# Don't tile over m
fill!(Cc, 0)
KernelTools.@tile (k,4,n,4) (Bt[n,k] = B[k,n]) begin
    KernelTools.@loophoist for n = 1:N,  k = 1:K, m = 1:M
        Cc[m,n] += A[m,k] * Bt[n,k]
    end
end
@test_approx_eq C Cc

# Image blur
A = rand(7,8)
blur_xy = similar(A)
interior_x, interior_y = 2:size(A,1)-1, 2:size(A,2)-1
KernelTools.@tile (x,4,y,4) (blur_y[x,y] = A[x,y-1]+A[x,y]+A[x,y+1]) begin
    for y = interior_y, x = interior_x
        blur_xy[x,y] = (1.0/9)*(blur_y[x-1,y] + blur_y[x,y] + blur_y[x+1,y])
    end
end
Ab = (A[interior_x, interior_y] + A[interior_x-1, interior_y] + A[interior_x+1, interior_y] +
      A[interior_x, interior_y-1] + A[interior_x-1, interior_y-1] + A[interior_x+1, interior_y-1] +
      A[interior_x, interior_y+1] + A[interior_x-1, interior_y+1] + A[interior_x+1, interior_y+1])/9
@test_approx_eq blur_xy[interior_x, interior_y] Ab
