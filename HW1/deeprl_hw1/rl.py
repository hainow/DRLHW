# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

import pdb


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    RUN QUESTION2.py TO TEST

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """

    value_func = np.zeros(env.nS)
    count = 0
    while True:
        # print("\t\tAt loop {} value of V[s]={}".format(count, V))
        delta = 0.
        value_func_new = np.zeros(env.nS)
        for s in range(env.nS):
            # print("\n\tAT state {}".format(s))
            a = policy[s]  # in this configuration, policy is always deterministic
            for p, s_next, r, done in env.P[s][a]:
                # print("\t\tfor action {} p={} s_next={} r={} done={}".format(a, p, s_next, r, done))
                value_func_new[s] += p * (r + gamma * value_func[s_next])
                # print("\t\told value = {} new value = {}".format(V[s], V_new[s]))

            delta = max(delta, np.abs(value_func[s] - value_func_new[s]))

        if (delta < tol) or (count > max_iterations):
            break

        # overwrite and go to next loop
        value_func = value_func_new
        count += 1

    return value_func, count


def evaluate_policy_inplace(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    RUN QUESTION2.py TO TEST

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """
    value_func = np.zeros(env.nS)
    count = 0
    while True:
        # print("\t\tAt loop {} value of V[s]={}".format(count, V))
        delta = 0.
        for s in range(env.nS):
            v_new = 0.
            a = policy[s]  # in this configuration, policy is always deterministic
            for p, s_next, r, done in env.P[s][a]:
                v_new += p * (r + gamma * value_func[s_next])

            delta = max(delta, np.abs(value_func[s] - v_new))
            value_func[s] = v_new
        if (delta < tol) or (count > max_iterations):
            break

        # overwrite and go to next loop
        count += 1

    return value_func, count #TODO: ask TA whether "count" here is ok?

def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    RUN QUESTION2.py TO TEST

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    nS = env.nS
    P = env.P
    optimal_policy = np.zeros(nS, dtype='int')  # optimal w.r.t this value_function given the 'env' only

    for s in range(nS):
        max_reward = -np.inf
        Ps = P[s]
        # scan through all possible actions provided by environment
        # and pick the one which yields the best cummulative reward
        for a in Ps:
            expected_value = 0.
            for p, s_next, r, done in Ps[a]:
                expected_value += p * (r + gamma * value_function[s_next])

            if max_reward < expected_value:
                max_reward = expected_value
                optimal_policy[s] = a

    return optimal_policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    RUN QUESTION2.py TO TEST

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    is_stable = True
    new_policy = value_function_to_policy(env, gamma, value_func)
    if not np.array_equal(policy, new_policy):
        is_stable = False

    return is_stable, new_policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    RUN QUESTION2.py TO TEST

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')

    impru_count = 1
    eval_count = 1
    while True:
        print("POLICY ITERATION - Loop no. {}".format(impru_count))
        # step 1: Evaluate Policy
        # value_func, c = evaluate_policy(env, gamma, policy, max_iterations, tol)
        value_func, c = evaluate_policy_inplace(env, gamma, policy, max_iterations, tol) # in-place is preferred (DP)
        eval_count += c
        print("\t\tNew V[s]={}".format(value_func))

        # step 2: Improve Policy
        is_stable, new_policy = improve_policy(env, gamma, value_func, policy)
        if is_stable or (impru_count > max_iterations):
            break

        # overwrite policy and evaluate again until stable
        policy = new_policy
        impru_count += 1

    return policy, value_func, impru_count, eval_count


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    RUN QUESTION2.py TO TEST

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    optimal_policy = np.zeros(env.nS)
    count = 1
    while True:
        delta = 0.
        # V_new = np.zeros(env.nS)
        for s in range(env.nS):
            v_new = value_func[s]
            max_return = -1.
            for a in env.P[s]:
                expected_value = 0.
                for p, s_next, r, done in env.P[s][a]:
                    expected_value += p * (r + gamma * value_func[s_next])
                if max_return < expected_value:
                    max_return = expected_value
                    optimal_policy[s] = a

            # update best value
            value_func[s] = max_return
            delta = max(delta, np.abs(value_func[s] - v_new))

        if (delta < tol) or (count > max_iterations):
            break
        # V_new = value_func
        count += 1

    return optimal_policy, value_func, count


# def print_policy(policy, action_names):
#     """Print the policy in human-readable format.
#
#     Parameters
#     ----------
#     policy: np.ndarray
#       Array of state to action number mappings
#     action_names: dict
#       Mapping of action numbers to characters representing the action.
#     """
#     str_policy = policy.astype('str')
#     for action_num, action_name in action_names.items():
#         np.place(str_policy, policy == action_num, action_name)
#
#     print(str_policy)
#
#
# def draw_policy(policy, action_names, map_size=0):
#     str_policy = policy.astype('str')
#     for action_num, action_name in action_names.items():
#         np.place(str_policy, policy == action_num, action_name)
#
#     print('\n\n')
#     for i in range( len(str_policy) / map_size ):
#         s = ''
#         for j in range(map_size):
#             s += str_policy[i * map_size + j]
#         print('\t\t{}'.format(s))
#
# def draw_value_func(value_func, map_size):
#     v = value_func.reshape(map_size, map_size)
#     plt.close()
#     plt.imshow(v, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.xticks(range(map_size))
#     plt.yticks(range(map_size))
#     plt.show()