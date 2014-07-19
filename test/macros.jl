import KernelTools
using Base.Test

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
@test stripexpr(ex) == stripexpr(:(for j = 1:size(A,2); _kt_1 = A[j]; for i = 1:size(A,1) s += _kt_1 end; A[j] = _kt_1; end))

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
# A[i]
stmnts = Any[]
offsetsyms, sizesyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:i]], [:i], [:ti])
@test stripexpr(stmnts[1]) == :($(offsetsyms[1]) = 1 - min(0, 1*(ti-1)+0))  # ultimately, this will evaluate to 1
@test stripexpr(stmnts[2]) == :($(sizesyms[1]) = $(offsetsyms[1]) + max(0, 1*(ti-1)+0))  # ultimately, this will evaluate to ti
# A[i-1]
stmnts = Any[]
offsetsyms, sizesyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:(i-1)]], [:i], [:ti])
@test stripexpr(stmnts[1]) == :($(offsetsyms[1]) = 1 - min(-1, 1*(ti-1)+(-1)))  # ultimately, this will evaluate to 2
@test stripexpr(stmnts[2]) == :($(sizesyms[1]) = $(offsetsyms[1]) + max(-1, 1*(ti-1)+(-1)))  # ultimately, this will evaluate to ti
# A[i-1] and A[i+1]
stmnts = Any[]
offsetsyms, sizesyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:(i-1)],[:(i+1)]], [:i], [:ti])
@test stripexpr(stmnts[1]) == :($(offsetsyms[1]) = 1 - min(-1, 1*(ti-1)+(-1), 1, 1*(ti-1)+1))  # ultimately, this will evaluate to 2
@test stripexpr(stmnts[2]) == :($(sizesyms[1]) = $(offsetsyms[1]) + max(-1, 1*(ti-1)+(-1), 1, 1*(ti-1)+1))  # ultimately, this will evaluate to ti+2
# A[i+1,j] and A[i,j+1]
stmnts = Any[]
offsetsyms, sizesyms = KernelTools.constructbounds!(stmnts, :A, Vector{Any}[[:(i+1),:j],[:i,:(j+1)]], [:i,:j], [:ti,:tj])
@test stripexpr(stmnts[1]) == :($(offsetsyms[1]) = 1 - min(1, 1*(ti-1)+1, 0, 1*(ti-1)+0))
@test stripexpr(stmnts[2]) == :($(sizesyms[1]) = $(offsetsyms[1]) + max(1, 1*(ti-1)+1, 0, 1*(ti-1)+0))
@test stripexpr(stmnts[3]) == :($(offsetsyms[2]) = 1 - min(0, 1*(tj-1)+0, 1, 1*(tj-1)+1))
@test stripexpr(stmnts[4]) == :($(sizesyms[2]) = $(offsetsyms[2]) + max(0, 1*(tj-1)+0, 1, 1*(tj-1)+1))

