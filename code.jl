using Plots:display
using Base:Float64
using LinearAlgebra
using DataFrames
using CSV
using Plots

function commutator(X::Array, Y::Array)
    return X * Y - Y * X
end

function action(X::Array)
    d = size(X)[1]
    N = size(X[1])[1]
    arr = zeros((N, N))
    for mu in 1:d
        for nu in 1:d
            comm = commutator(X[mu], X[nu])
            arr += comm * comm
        end
    end
    return -N * real(tr(arr)) / 4
end

function metropolize!(XOld::Array, eps::Float64, iters::Int64, gap::Int64, isTherm::Bool)
    XNew = deepcopy(XOld)
    d = size(XOld)[1]
    N = size(XOld[1])[1]
    accept = 0
    if (!isTherm)
        df = DataFrame(Iteration=[], Acceptance=[], Action=[])
    end
    for t in 1:iters
        lambda = rand(1:1:d)
        indexR = rand(1:1:N)
        indexC = rand(1:1:N)
        if (indexR == indexC)
            noise = 2 * eps * (rand(Float64) - 0.5)
        else
            noise = 2 * eps * (rand(ComplexF64) - 0.5 - 0.5im)
        end
        XNew[lambda][indexR, indexC] = XOld[lambda][indexR, indexC] + noise
        XNew[lambda][indexC, indexR] = conj(XNew[lambda][indexR, indexC])
        @assert XNew[lambda] == XNew[lambda]'
        newAction = action(XNew)
        oldAction = action(XOld)
        ΔS = newAction - oldAction
        u = rand(Float64)
        if (ΔS <= 0 || exp(-ΔS) >= u)
            oldAction = newAction
            XOld[lambda][indexR, indexC] = XNew[lambda][indexR, indexC]
            XOld[lambda][indexC, indexR] = XNew[lambda][indexC, indexR]
            accept += 1
        else
            newAction = oldAction
            XNew[lambda][indexR, indexC] = XOld[lambda][indexR, indexC]
            XNew[lambda][indexC, indexR] = XOld[lambda][indexC, indexR]
        end
        @assert XNew[lambda] == XNew[lambda]'
        @assert XOld[lambda] == XOld[lambda]'
        if (t % gap == 0)
            if (!isTherm)
                push!(df, [t, accept * 100 / t, newAction])
            end
        end
    end
    if (!isTherm)
        return df
    end
end

function hermitianMatrix(N::Int64)
    X = zeros(ComplexF64, (10, 10))
    for i in 1:N, j in 1:N
        if (j > i)
            X[i,j] = rand(ComplexF64)
        elseif (i == j)
            X[i,j] = rand(Float64) + 0.0im
        else
            X[i,j] = conj(X[j,i])
        end
    end
    @assert X == X'
    return X
end

function drive(d::Int64, N::Int64, eps::Float64, gap::Int64, therm::Int64, sweeps::Int64)
    X = Array
    for i in 1:d
        arr = hermitianMatrix(N)
        if (i == 1)
            X = [arr]
        else
            push!(X, arr)
        end
    end
    metropolize!(X, eps, therm, gap, true)
    df = metropolize!(X, eps, sweeps, gap, false)
    println(sum(df."Action")/length(df."Action"))
    CSV.write("data.csv", df)
    p1 = plot(df."Iteration", df."Acceptance", title="Acceptance", label="d=4, N=10")
    p2 = plot(df."Iteration", df."Action", title="Action", label="d=4, N=10")
    plot(p1, p2, layout=(2, 1))
    savefig("Plots.png")
end

drive(Int64(4), Int64(10), Float64(0.1), Int64(100), Int64(5*1e4), Int64(1e6))
