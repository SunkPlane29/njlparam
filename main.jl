#NOTE: References:
# S. P. Klevansky, The Nambu--Jona-Lasinio model of quantum chromodynamics, Rev. Mod. Phys. 64, 649 – Published 1 July, 1992
# Michael Buballa, NJL-model analysis of dense quark matter, Physics Reports, Volume 407, Issues 4–6, 2005
# Dyana Cristine Duarte, ESTRUTURA DE FASES DA MATÉRIA DE QUARKS QUENTE, DENSA E MAGNETIZADA NO MODELO DE NAMBU–JONA-LASINIO

using NonlinearSolve
using StatsPlots
using Serialization
using NestedSamplers
using Revise
using AbstractMCMC
using StaticArrays
using Distributions
using DataFrames
using CSV

includet("integrals.jl")
includet("param.jl")

function deterministic_params()
    fpi = 0.0924
    mpi = 0.135
    Lamb = 0.64

    solvec = zeros(5)
    getparam_Lambin(Lamb, fpi, mpi, solvec)
end

begin
    meanfpi = 0.0924
    sigfpi = 2e-3

    meanmpi = 0.1349768
    sigmpi = 5e-7

    cond_lb = 0.190 
    cond_ub = 0.260

    fixedLamb = 0.64
end

function loglike(theta::AbstractVector)
    Lambda, G, mc = theta  

    M = solvegap(Lambda, G, mc)
    mpi = getmpi(M, Lambda, G, mc)
    fpi = fpieq(M, Lambda)
    cond = -cbrt(quarkcond(M, Lambda, G, mc))
    
    #condensate uniform likelihood works to truncate the likelihood
    if cond < cond_lb || cond > cond_ub
        return -Inf
    end

    total_logL = 0.0

    total_logL += logpdf(Normal(meanmpi, sigmpi), mpi)
    total_logL += logpdf(Normal(meanfpi, sigfpi), fpi)

    return total_logL
end

function priortransform(u::AbstractVector)
    theta = zeros(eltype(u), length(u))
    L_min = 0.45
    L_max = 0.85
    G_min = 1.5/L_max^2
    G_max = 2.5/L_min^2
    mc_min = 0.003
    mc_max = 0.007

    theta[1] = L_min + (L_max - L_min) * u[1]
    theta[2] = G_min + (G_max - G_min) * u[2]
    theta[3] = mc_min + (mc_max - mc_min) * u[3]

    return theta
end

function nestedsampling_chain()
    ndims = 3
    nlive = 20_000
    sampler = Nested(ndims, nlive)
    model = NestedModel(loglike, priortransform)
    
    println("Starting nested sampling...")
    chain = NestedSamplers.sample(model, sampler, dlogz=0.1) # Aumentei um pouco o dlogz 
    println("Nested sampling completed.")

    serialize("njlparameters_chain.jls", chain)
end

function plotchain()
    chain = deserialize("njlparameters_chain.jls")
    plot(chain[1])
    savefig("njlparameters_chain.png")
end

function chain_to_csv(burnin=1)
    df = DataFrame(deserialize("njlparameters_chain.jls")[1])
    df = df[burnin:end, :]
    rename!(df, [:iter, :chain, :Lamb, :GL2, :mc, :w])
    CSV.write("njlparameters_chain.csv", df)
end
