macroexpand_jl(io::IO, ex::Expr) = Base.show_unquoted(io, macroexpand(ex))

function Base.show_block(io::IO,head,arg,block,i::Int)
    if isa(arg, Expr) && arg.head == :block
        Base.show_block(io,head,arg.args,block,i)
    else
        Base.show_block(io,head,{arg},block,i)
    end
end
