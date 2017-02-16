import gym
import time

import numpy as np
import matplotlib.pyplot as plt

import deeprl_hw1.queue_envs as queue_envs
import deeprl_hw1.rl as rl

def print_transition(env):
    np.random.choice([1, 2, 3])
    print("\nTransition policy of this env:")
    for s in env.states:
        # if s == (0, 0, 0, 0):
        print('State {} ->'.format(s))
        for a in range(env.nA):
            print ('\taction {}'.format(a))
            for p, next_state, r, done in env.P[s][a]:
                print ('\t\t ===> p={} s_next={} r={} done={}'.
                       format(p, next_state, r, done))


def run_random_policy(env, max_steps=100):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0.
    num_steps = 1
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal or num_steps > max_steps:
            break

        time.sleep(.5)

    return total_reward


def queue_evaluate_policy_inplace(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

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
    # value_func = np.zeros(env.nS)
    value_func = {}
    for s in env.states:
        value_func[s] = 0
    count = 0
    while True:
        # print("\t\tAt loop {} value of V[s]={}".format(count, V))
        delta = 0.
        for s in env.states:
            # print("\n\tAT state {}".format(s))
            # base case: check whether s is a terminal state
            # if s in terminals:
            #     continue

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

    return value_func, count


def queue_value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

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
    P = env.P
    optimal_policy = {} # mapping state -> optimal action
    for s in env.states:
        optimal_policy[s] = 0

    for s in env.states:
        max_reward = -1
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


def queue_improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

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
    new_policy = queue_value_function_to_policy(env, gamma, value_func)
    if not np.array_equal(policy, new_policy):
        is_stable = False

    return is_stable, new_policy


def queue_policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3, verbose=False):
    """Runs policy iteration.

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
    policy = {}
    for s in env.states:
        policy[s] = 0

    impru_count = 1
    eval_count = 1
    while True:
        if verbose:
            print("POLICY ITERATION - Loop no. {}".format(impru_count))
        # step 1: Evaluate Policy
        # value_func, c = evaluate_policy(env, gamma, policy, max_iterations, tol)
        value_func, c = queue_evaluate_policy_inplace(env, gamma, policy, max_iterations, tol)
        eval_count += c
        if verbose:
            print("\t\tNew V[s]={}".format(value_func))

        # step 2: Improve Policy
        is_stable, new_policy = queue_improve_policy(env, gamma, value_func, policy)
        if is_stable or (impru_count > max_iterations):
            break

        # overwrite policy and evaluate again until stable
        policy = new_policy
        impru_count += 1

    return policy, value_func, impru_count, eval_count


def queue_value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

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
    value_func, optimal_policy = {}, {}
    for s in env.states:
        value_func[s], optimal_policy[s] = 0, 0
    count = 1
    while True:
        delta = 0.
        for s in env.states:
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
        count += 1

    return optimal_policy, value_func, count


def main():
    ''' TEST DRIVER FOR Q3'''

    env = gym.make('Queue-1-v0')
    print_transition(env)

    # Run a random policy for 10 steps
    iters = 10
    total_reward = run_random_policy(env, max_steps=iters)
    print("Total reward = {} after {} iters, average = {}".format(total_reward, iters, total_reward / float(iters)))

    # t1 = time.time()
    # policy = {}
    # for s in env.states:
    #     policy[s] = 0
    # value_func, eval_count = queue_evaluate_policy_inplace(env=env, gamma=.9, policy=policy)
    # print value_func, eval_count

    # WARNING: very long since there are many states
    # t1 = time.time()
    # policy, value_func, impru_count, eval_count = queue_policy_iteration(env=env, gamma=0.9, verbose=False)
    # print policy
    # print("\n\n\na. Time taken for POLICY iteration for question 2.a is {} seconds".format(time.time() - t1))
    # print("a. Policy improvement takes {} steps, policy evaluation takes {} steps".format(impru_count, eval_count))

    # MUCH FASTER THAN POLICY ITERATION
    # t2 = time.time()
    # optimal_policy, value_func, count = queue_value_iteration(env, gamma=0.9)
    # print("\n\n\nTime taken for VALUE iteration for question 2.a is {} seconds".format(time.time() - t2))
    # print("Total iterations is {}".format(count))
    # print("\n\nOptimal policy for VALUE iteration is: {}".format(optimal_policy))
    # print("\n\nOptimal VALUES for each action is {}".format(value_func))

if __name__ == "__main__":
    main()