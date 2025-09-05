Nf = 2
Nc = 3

# Verificado por Arthur EBP 04/09/2025
function fpiint(M, Lamb)
    1/8π^2 * (asinh(Lamb/M) - Lamb/sqrt(Lamb^2 + M^2)) # ok 
end

function gapsys!(du, u, p)
    du[1] = Mgap(u[1], p[1], p[2], p[3])
end

function solvegap(Lamb, G, mc)
    u0 = SA[0.32]
    prob = NonlinearProblem(gapsys!, u0, SA[Lamb, G, mc])
    return solve(prob, SimpleNewtonRaphson()).u[1]
end

function mpiint(q, M, Lamb)
    # println("q = $q, M = $M, Lamb = $Lamb")
    @assert q < 2M "q must be lesser than 2M" # ok 
    1/8π^2 * (1/2 * log((Lamb + sqrt(Lamb^2 + M^2))^2/M^2) - sqrt(4M^2/q^2 - 1) * atan(Lamb/(sqrt(Lamb^2 + M^2)*sqrt(4M^2/q^2 - 1))))
end

function fpieq(M, Lamb)
    # println("M = $M, Lamb = $Lamb")
    if isnan(M) || M < 0.0
        return 0.0
    end

    @assert fpiint(M, Lamb) > 0 "fpiint must be positive"
    sqrt(4*Nc*M^2*fpiint(M, Lamb))
end

function gapint(M, Lamb)
    1/4π^2 * (Lamb*sqrt(Lamb^2 + M^2) - M^2/2 * log((Lamb + sqrt(Lamb^2 + M^2))^2/M^2))
end

# Vanishing chemical potential fermi-dirac distribution integral 
function gap_Tint(M,T)
    1/2π^2*quadgk(k->k^2/(1+exp(sqrt(k^2+M^2)/T)),0.0,Inf64,maxevals=16)[1]
end

function Mgap(M, Lamb, G, mc)
    M - mc - 4G*Nf*Nc*M*gapint(M, Lamb)
end

# Usando despacho múltiplo
function Mgap(M, Lamb, G, mc,T)
    M - mc - 4G*Nf*Nc*M*(gapint(M, Lamb)-gap_Tint(M,T))
end

function mpieq(mpi, M, Lamb, G, mc)
    mc/M - 4G*Nf*Nc*mpi^2*mpiint(mpi, M, Lamb)
end

function mpisys!(du, u, p)
    du[1] = mpieq(u[1], p[1], p[2], p[3], p[4])
end

function getmpi(M, Lamb, G, mc)
    if M < 0.135 || isnan(M)
        return 0.0
    end

    u0 = SA[0.135]
    prob = NonlinearProblem(mpisys!, u0, SA[M, Lamb, G, mc])
    sol = solve(prob, SimpleNewtonRaphson())
    return sol.u[1]
end

function quarkcond(M, Lamb, G, mc)
    -(M - mc)/4G
end