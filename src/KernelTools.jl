module KernelTools

export @loophoist, @tile, @test_looporder

if VERSION.minor < 3
    using Cartesian
else
    using Base.Cartesian
end

include("edge.jl")
include("macros.jl")

end # module
