Nf = 2
Nc = 3

# Verificado por Arthur EBP 04/09/2025
function fpiint(M, Lamb)
    1/8π^2 * (asinh(Lamb/M) - Lamb/sqrt(Lamb^2 + M^2)) # ok 
end

function gapsys!(du, u, p)
    du[1] = Mgap(u[1], p...) # Aha! 
end

function solvegap(Lamb, G, mc,T)
    u0 = [Lamb] # Geralmente no caso \mu=0 começar com M ~ Lambda é um bom chute
    prob = NonlinearProblem(gapsys!, u0, SA[Lamb, G, mc,T]) # 
    return solve(prob, FixedPointAccelerationJL()).u[1] #
    # Como estamos apenas buscando a T_pc, não precisamos
    # mais de um ponto de partida, pois a solução é única # e suave. - Arthur EBP 08/09/2025
    # O nonlinearproblem não tá funcionando com temperatura não sei pq. 
end

function solvegap(Lamb, G, mc)
    u0 = SA[Lamb]
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
    # Temos duas vezes o mesmo termo, uma que a \mu = 0 
    # a flutuação de partículas e a de antipartículas são iguais
    # println(M,' ',T)
    1/2π^2*quadgk(k->k^2/(1.0+exp(sqrt(k^2+M^2)/T)),0.0,Inf,maxevals=16)[1]
    # O problema está nessa integral 
    # não sei o que está acontecendo de errado
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

function get_Tpc(Lamb,G,mc;steps=10,intervals=5)
    Tmin = 0.0 
    Tmax = Lamb
    for i in 1:steps
        Ts = range(Tmin, Tmax, length=intervals)
        M = -solvegap.(Lamb,G,mc,Ts)
        # indice do maior degrau
        diffs = abs.(diff(M))
        maxind = argmax(diffs)
        Tmin = Ts[maxind]
        Tmax = Ts[maxind+1]
    end
    return (Tmin + Tmax)/2
end

function quarkcond(M, Lamb, G, mc)
    -(M - mc)/4G
end