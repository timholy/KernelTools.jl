module KernelTools

if VERSION.minor < 3
    using Cartesian
else
    using Base.Cartesian
end

include("edge.jl")

end # module
