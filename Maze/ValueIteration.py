import numpy as np

from Maze.envs.gridworld import GridworldEnv

"""
    4 x 4 的环境下, 左上和右下为出口, 其中价值为: T = 0, 其他 = -1
    T   O   O   O
    O   X   O   O
    O   O   O   O
    O   O   O   T
"""
env = GridworldEnv()


def value_interation(env, theta=0.0001, discount_factor=1.0):

    def one_step_lookahead(state, v):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * v[next_state])
        return A

    v = np.zeros(env.nS)

    while True:
        delta = 0

        for s in range(env.nS):
            A = one_step_lookahead(s, v)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - v[s]))
            v[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        A = one_step_lookahead(s, v)
        best_action = np.max(A)
        policy[s, int(best_action)] = 1.0
    return policy, v


policy, v = value_interation(env)
print('Policy Probability Distribution\n', policy)
print('Reshaped Grid Policy (0=up, 1=right, 2=dow, 3-left)：\n', np.reshape(np.argmax(policy, axis=1), env.shape))
