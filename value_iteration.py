
S = ["B", "G"]  # state
A = [0, 1]  # action
nu = 0.05
beta = 0.33


def P(s_next, s, a):
    if s == "G" and s_next == "G" and a == 1:
        return 0.3
    elif s == "G" and s_next == "G" and a == 0:
        return 0.7
    elif s == "B" and s_next == "G" and a == 1:
        return 0.5
    elif s == "B" and s_next == "G" and a == 0:
        return 0.9
    elif s == "G" and s_next == "B" and a == 1:
        return 0.7
    elif s == "G" and s_next == "B" and a == 0:
        return 0.3
    elif s == "B" and s_next == "B" and a == 1:
        return 0.5
    # elif s == "B" and s_next == "B" and a == 0:
    # return 0.1
    else:
        return 0.1


def R(s, a):
    if s == "G" and a == 1:  # win
        return beta*(-1 + nu * a)
    else:
        return beta*(nu * a)


def value_iteration(S, A, P, R):
    V = {s: 0 for s in S}
    optimal_policy = {s: 0 for s in S}
    while True:
        oldV = V.copy()

        for s in S:
            Q = {}
            for a in A:
                Q[a] = R(s, a) + beta * sum(P(s_next, s, a) * oldV[s_next] for s_next in S)

            V[s] = min(Q.values())
            optimal_policy[s] = min(Q, key=Q.get)

        if all(oldV[s] == V[s] for s in S):
            break

    return V, optimal_policy


V, optimal_policy = value_iteration(S, A, P, R)
print(V)
print(optimal_policy)

