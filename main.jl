begin
    using NonlinearSolve
    using StaticArrays
end

begin
    include("integrals.jl")
    include("param.jl")
end

fpieq(0.0924, 0.3, 0.64)
mpieq(0.135, 0.3, 0.64, 1.97653/0.64^2, 0.005)

let
    fpi = 0.0924
    mpi = 0.135
    Lamb = 0.64

    solvec = @MArray zeros(5)
    getparam_Lambin(Lamb, fpi, mpi, solvec)

    println(solvec)
end