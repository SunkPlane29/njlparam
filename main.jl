begin
    using NonlinearSolve
    using StaticArrays
    using Turing
    using StatsPlots
end

begin
    include("integrals.jl")
    include("param.jl")
end

let
    fpi = 0.0924
    mpi = 0.135
    Lamb = 0.64

    solvec = @MArray zeros(5)
    getparam_Lambin(Lamb, fpi, mpi, solvec)
    solvec
end

begin
    meanfpi = 0.0924
    sigfpi = 2e-3

    meanmpi = 0.1349768
    sigmpi = 5e-7

    acbcond = 0.190 
    bcbcond = 0.260
    sigcond = sqrt(1/12 * (0.260 - 0.190)^2)

    fixedLamb = 0.64
end

@model function njlparams(fpi, mpi, cond)
    # Lamb ~ Uniform(0.580, 0.700)
    Lamb = fixedLamb
    G ~ Uniform(1.5/0.700^2, 2.5/0.580^2)
    mc ~ Uniform(0.004, 0.006)

    Mmodel = solvegap(Lamb, G, mc)
    fpimodel = fpieq(Mmodel, Lamb)
    mpimodel = getmpi(Mmodel, Lamb, G, mc)
    condmodel = -cbrt(quarkcond(Mmodel, Lamb, G, mc))

    fpi ~ Normal(fpimodel, sigfpi)
    mpi ~ Normal(mpimodel, sigmpi)
    cond ~ Normal(condmodel, sigcond)
end

# You have to run a lot of times until you stop getting errors, I think this has something to do with the
# initial values of the parameters
begin
    fpidist = Normal(meanfpi, sigfpi)
    mpidist = Normal(meanmpi, sigmpi)
    conddist = Uniform(acbcond, bcbcond)

    fpivals = rand(fpidist, 1000)
    mpivals = rand(mpidist, 1000)
    condvals = rand(conddist, 1000)

    model = njlparams(fpivals, mpivals, condvals)

    chain = sample(model, NUTS(0.65), 4000)
end

begin
    x = chain[:G][:,1]
    y = chain[:mc][:,1]

    #I want to only take x values higher than 5.18 and yvalues higher than 5.17,
    #and y mathcing the x values
    x_filtered = x[x .> 5.18]
    y_filtered = y[x .> 5.18]

    marginalkde(x_filtered, y_filtered*1e3, levels=4, xlabel=raw"$G$ [GeV$^{-2}$]", ylabel=raw"$m$ [MeV]")
end