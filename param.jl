function mpifpisys_Lambin!(du, u, p)
    du[1] = Mgap(u[1], p[1], u[2], u[3])
    du[1] = p[2] - fpieq(u[1], p[1])
    du[2] = mpieq(p[3], u[1], p[1], u[2], u[3])
end

function getparam_Lambin(Lamb, fpi, mpi, solvec)
    u0 = SA[0.32, 2.1/Lamb^2, 0.005]
    p = SA[Lamb, fpi, mpi]

    prob = NonlinearProblem(mpifpisys_Lambin!, u0, p)
    sol = solve(prob, SimpleKlement(), abstol=1e-8)

    solvec[1] = sol.u[1]
    solvec[2] = Lamb
    solvec[3] = sol.u[2]
    solvec[4] = sol.u[3]
    solvec[5] = -cbrt(quarkcond(sol.u[1], Lamb, sol.u[2], sol.u[3]))
end