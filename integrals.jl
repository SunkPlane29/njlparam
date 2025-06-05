Nf = 2
Nc = 3

function fpiint(M, Lamb)
    1/8π^2 * (asinh(Lamb/M) - Lamb/sqrt(Lamb^2 + M^2)) 
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
    1/8π^2 * (log((Lamb + sqrt(Lamb^2 + M^2))/M) - sqrt(4M^2/q^2 - 1) * atan(Lamb/(sqrt(Lamb^2 + M^2)*sqrt(4M^2/q^2 - 1))))
end

function fpieq(fpi, M, Lamb)
    fpi^2 - 4*Nc*M^2*fpiint(M, Lamb)
end

function gapint(M, Lamb)
    1/4π^2 * (Lamb*sqrt(Lamb^2 + M^2) - M^2/2 * log((Lamb + sqrt(Lamb^2 + M^2))^2/M^2))
end

function Mgap(M, Lamb, G, mc)
    M - mc - 4G*Nf*Nc*M*gapint(M, Lamb)
end

function mpieq(mpi, M, Lamb, G, mc)
    println(mc/M)
    println(- 4G*Nf*Nc*mpi^2*mpiint(mpi, M, Lamb))
    mc/M - 4G*Nf*Nc*mpi^2*mpiint(mpi, M, Lamb)
end

function quarkcond(M, Lamb, G, mc)
    -(M - mc)/4G
end