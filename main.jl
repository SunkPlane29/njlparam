#NOTE: References:
# S. P. Klevansky, The Nambu--Jona-Lasinio model of quantum chromodynamics, Rev. Mod. Phys. 64, 649 – Published 1 July, 1992
# Michael Buballa, NJL-model analysis of dense quark matter, Physics Reports, Volume 407, Issues 4–6, 2005
# Dyana Cristine Duarte, ESTRUTURA DE FASES DA MATÉRIA DE QUARKS QUENTE, DENSA E MAGNETIZADA NO MODELO DE NAMBU–JONA-LASINIO

# Reference to the Pseudocritical Temperature from Lattice QCD
# From HotQCD Collaboration, A. Bazavov et al.
# https://arxiv.org/pdf/1812.08235
# Tpc = 156.5 ± 1.5 MeV
# TODO: 
# [x] Implementar uma função que calcule o Tpc a partir dos parâmetros do modelo NJL 

# Adaptando pro meu workflow Arthur EBP 08/09/2025
begin
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
    using QuadGK
    using FixedPointAcceleration
    using Plots

    includet("integrals.jl")
    includet("param.jl")
end 

function deterministic_params(;Lamb = 0.64)
    fpi = 0.0924
    mpi = 0.135

    solvec = zeros(6)
    getparam_Lambin(Lamb, fpi, mpi, solvec)
end

deterministic_params()

begin
    meanfpi = 0.0924
    sigfpi = 5e-3

    meanmpi = 0.1349768
    sigmpi = 5e-7

    cond_lb = 0.190 
    cond_ub = 0.260
    cond_lattice = 0.231 # valor central da lattice
    sigcond = 0.008 # erro estimado

    Lattice_Tpc = 0.1565
    Lattice_Tpc_err = 0.0015

    fixedLamb = 0.64
end

function get_values(Lambda, G, mc)
    M = solvegap(Lambda, G, mc)
    mpi = getmpi(M, Lambda, G, mc)
    fpi = fpieq(M, Lambda)
    cond = -cbrt(quarkcond(M, Lambda, G, mc))
    T_pc = get_Tpc(Lambda, G, mc)

    return M, mpi, fpi, cond, T_pc
end

# plot(LinRange(1.0,3.0,200),x->get_values(0.45,x/(0.45^2),0.003)[end])
get_values(0.45,1.6/(0.45^2),0.003)

function loglike(theta::AbstractVector)
    Lambda, G, mc = theta  
    G = G / Lambda^2 # eu entendo melhor assim :D
    M, mpi, fpi, cond, T_pc = get_values(Lambda, G, mc)

    # condensate uniform likelihood works to truncate the likelihood
    if cond < cond_lb || cond > cond_ub
        return -Inf
    end

    total_logL = 0.0

    # total_logL += logpdf(Normal(cond_lattice, sigcond), cond)
    total_logL += logpdf(Normal(meanmpi, sigmpi), mpi)
    total_logL += logpdf(Normal(meanfpi, sigfpi), fpi)
    total_logL += logpdf(Normal(Lattice_Tpc, Lattice_Tpc_err), T_pc)

    return total_logL
end

function priortransform(u::AbstractVector)
    theta = zeros(eltype(u), length(u))
    L_min = 0.4
    L_max = 0.7
    GL2_min = 1.2
    GL2_max = 3.5
    mc_min = 0.002
    mc_max = 0.006

    theta[1] = L_min + (L_max - L_min) * u[1]
    theta[2] = GL2_min + (GL2_max - GL2_min) * u[2]
    theta[3] = mc_min + (mc_max - mc_min) * u[3]

    return theta
end

function nestedsampling_chain()
    ndims = 3
    nlive = 5_000
    sampler = Nested(ndims, nlive)
    model = NestedModel(loglike, priortransform)
    
    println("Starting nested sampling...")
    chain = NestedSamplers.sample(model, sampler, dlogz=2e1)
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
    # df[!,"T_pc"] = [get_Tpc(row.Lamb, row.GL2 / row.Lamb^2, row.mc) for row in eachrow(df)]
    # df["Cond"] = [-cbrt(quarkcond(row.Lamb, row.GL2 / row.Lamb^2, row.mc)) for row in eachrow(df)]
    # df["mpi"] = [getmpi(solvegap(row.Lamb, row.GL2 / row.Lamb^2, row.mc), row.Lamb, row.GL2 / row.Lamb^2, row.mc) for row in eachrow(df)]
    # df["fpi"] = [fpieq(solvegap(row.Lamb, row.GL2 / row.Lamb^2, row.mc), row.Lamb) for row in eachrow(df)]
    CSV.write("njlparameters_chain.csv", df)
end
