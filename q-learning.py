import numpy as np
import random
import itertools
import pandas as pd

iterations = 500
c1, c2, c3, c4 = 0, 0, 0, 0
xpoints = np.zeros(iterations + 1)
ypoints = np.zeros(iterations + 1)
S = ["B", "G"]  # state
A = [0, 1]  # action
nu = 0.95
beta = 0.33
epsilon = 0.5
q = np.zeros([len(S), len(A)])
q_table = pd.DataFrame(q, index=S, columns=A)

print("Initial Q-Table: \n", q_table)


def R(s, a):
    if s == "G" and a == 1:  # win
        rv = beta*(-1 + nu*float(a))
        return rv
    else:
        rv = beta*(nu*float(a))
        return rv


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


def cost(s, s_next, action):
    Prob = [0, 0]
    Rew = [0, 0]
    sn = [0, 0]
    done = False

    for x in A:
        Prob[x] = P(s_next, s, x)
        Rew[x] = R(s, x)

    pair = zip(Prob, itertools.repeat(s_next), Rew, itertools.repeat(done))
    reward = {}
    for a, b in zip(A, pair):
        reward[a] = b
    return reward[action]


def alpha(s, a):
    cnd = [c1, c2, c3, c4]
    if s == "G" and a == 1:
        j = 0
    elif s == "G" and a == 0:
        j = 1
    elif s == "B" and a == 1:
        j = 2
    else:
        j = 3

    al = 1/(1 + cnd[j])
    return al


state = random.choice(S)
next_state = random.choice(S)

for i in range(1, iterations + 1):

        if random.uniform(0, 1) < epsilon:      # Randomly explore action space
            action = random.choice(A)
            next_state = random.choice(S)
        else:                                   # use learned values
            max_V = min(q_table.loc[state])
            action = (q_table == max_V).idxmax(axis=1)[0]

        probability, next_state, reward, done = cost(state, next_state, action)

        s = q_table[action]

        current_value = s[state]
        next_max = min(q_table.loc[next_state])

        new_value = (1-alpha(state, action))*current_value + alpha(state, action) * (reward + beta * (next_max - current_value))
        new_s = q_table[action]
        new_s[state] = new_value
        state = next_state
        i += 1

        if state == "G" and action == 1:
            c1 += 1
        elif state == "G" and action == 0:
            c2 += 1
        elif state == "B" and action == 1:
            c3 += 1
        elif state == "B" and action == 0:
            c4 += 1


print("Final Q-Table: \n", q_table)
