S = ["B", "G"]  # state
A = [0, 1]  # action
nu = 0.95
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


def policy_iteration(S, A, P, R):
    policy = {s: A[0] for s in S}

    while True:
        old_policy = policy.copy()

        V = policy_evaluation(policy, S)
        policy = policy_improvement(V, S, A)

        if all (old_policy[s] == policy[s] for s in S):
            break
    return policy


def policy_evaluation(policy, S):
    V = {s: 0 for s in S}

    while True:
        oldV = V.copy()

        for s in S:
            a = policy[s]
            V[s] = R(s, a) + beta* sum(P(s_next, s, a) * oldV[s_next] for s_next in S)

        if all(oldV[s] == V[s] for s in S):
            break
    return V


def policy_improvement(V, S, A):
    policy = {s: A[0] for s in S}
    for s in S:
        Q={}
        for a in A:
            Q[a] = R(s, a) + beta * sum(P(s_next, s, a) * V[s_next] for s_next in S)

        policy[s] = min(Q, key=Q.get)

    return policy


optimal_policy = policy_iteration(S, A, P, R)
print(optimal_policy)

