import gym
import time
import pdb

import numpy as np
import matplotlib.pyplot as plt

from gym import wrappers

import deeprl_hw1.lake_envs as lake_envs  # command line import version
import deeprl_hw1.rl as rl  # command line import version


def print_transition(env):
    print("\nTransition policy of this env:")
    for s in range(env.nS):
        print('State {} ->'.format(s))
        for a in range(env.nA):
            print ('\taction {}'.format(a))
            for p, s_next, r, done in env.P[s][a]:
                print ('\t\t ===> p={} s_next={} r={} done={}'.format(p, s_next, r, done))


def print_initial_distribution(env):
    print("\nInitial distribution of states (with total of {} states) is: ".format(env.nS))
    print(env.isd)


def draw_policy_console(policy, action_names, map_size=0):
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
    for i in range(len(str_policy) / map_size):
        s = ''
        for j in range(map_size):
            s += str_policy[i * map_size + j]
        print('\t\t{}'.format(s))


def draw_str_policy_console(str_policy, map_size=0):
    for i in range(len(str_policy) / map_size):
        s = ''
        for j in range(map_size):
            s += str_policy[i * map_size + j]
        print('\t\t{}'.format(s))


def draw_value_func_heatmap(value_func, map_size=0):
    v = value_func.reshape(map_size, map_size)
    # plt.imshow(v, cmap='hot', interpolation='nearest')
    plt.imshow(v, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(map_size))
    plt.yticks(range(map_size))
    plt.show()


def convert_values_to_optimal_policy(env, value_function):
    """ similar to rl.value_function_to_policy() except for eliminating gamma"""
    nS = env.nS
    P = env.P
    optimal_policy = np.zeros(nS, dtype='int')
    for s in range(nS):
        max_reward = -1
        Ps = P[s]
        # scan through all possible actions provided by environment
        # and pick the one which yields the best expected reward
        for a in Ps:
            expected_value = 0.
            for p, s_next, r, done in Ps[a]:
                expected_value += p * (r + value_function[s_next])  # TODO: solve the second-to-goal state
            if max_reward < expected_value:
                max_reward = expected_value
                optimal_policy[s] = a

    return optimal_policy


def execute_optimal_policy_cummulative(env, policy, gamma=.9, verbose=False):
    total_reward = 0.
    num_steps = 0
    current_state = env.reset()
    while True:
        next_state, reward, is_terminal, debug_info = env.step(policy[current_state])
        if verbose:
            print("Start {} action {} -> next state {}".format(current_state, policy[current_state], next_state))
        total_reward += (gamma ** num_steps) * reward
        num_steps += 1

        if is_terminal:
            break

        current_state = next_state

    return total_reward, num_steps


############# CONSTANT ############
# LEFT = 0    DOWN = 1    RIGHT = 2   UP = 3
action_names = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}


def question_2_a():
    for env_name, map_size in [('Deterministic-4x4-FrozenLake-v0', 4), ('Deterministic-8x8-FrozenLake-v0', 8)]:
        env = gym.make(env_name)
        print("\n\n\n==============================================================================================\n")
        print("Processing env {}".format(env_name))

        # Question 2 - Part a
        # print_transition(env)
        # print_initial_distribution(env)

        # a
        t1 = time.time()
        policy, value_func1, impru_count, eval_count = rl.policy_iteration(env=env, gamma=0.9)
        print policy
        print("\n\n\na. Time taken for POLICY iteration for question 2.a is {} seconds".format(time.time() - t1))
        print("a. Policy improvement takes {} steps, policy evaluation takes {} steps".format(impru_count, eval_count))

        # b
        print("\n\nOptimal policy for POLICY iteration is: ")
        draw_policy_console(policy, action_names, map_size)

        # c
        print("\nc. Value function for POLICY iteration = {}".format(value_func1))
        draw_value_func_heatmap(value_func1, map_size)

        # d
        t2 = time.time()
        optimal_policy, value_func2, count = rl.value_iteration(env, gamma=0.9)
        print("\n\n\nd. Time taken for VALUE iteration for question 2.a is {} seconds".format(time.time() - t2))
        print("d. Total iterations is {}".format(count))
        print("\n\nOptimal policy for VALUE iteration is: ")
        draw_policy_console(optimal_policy, action_names, map_size)

        # e
        print("\ne. Value function for VALUE iteration = {}".format(value_func2))
        draw_value_func_heatmap(value_func2, map_size)

        # f
        optimal_policy_converted = rl.value_function_to_policy(env, gamma=0.9, value_function=value_func2)
        print("\nf. Convert optimal values to optimal policy")
        draw_policy_console(optimal_policy_converted, action_names, map_size)
        print(optimal_policy_converted == optimal_policy)

        # g
        total_reward, num_steps = execute_optimal_policy_cummulative(env, optimal_policy)
        print("g. Total CUMMULATIVE DISCOUNTED reward = {}, total steps = {}".format(total_reward, num_steps))
        print("==============================================================================================\n\n\n")


def question_2_b():
    for env_name, map_size in [('Stochastic-4x4-FrozenLake-v0', 4), ('Stochastic-8x8-FrozenLake-v0', 8)]:
        env = gym.make(env_name)
        print("\n\n\n==============================================================================================\n")
        print("Processing env {}".format(env_name))

        # a
        t1 = time.time()
        optimal_policy, value_func, count = rl.value_iteration(env, gamma=0.9)
        print("\n\n\na. Time taken for VALUE iteration for question 2.b is {} seconds".format(time.time() - t1))
        print("a. Total iterations is {}".format(count))
        print("\n\nOptimal policy for VALUE iteration is: ")
        draw_policy_console(optimal_policy, action_names, map_size)

        # b
        print("\nc. Value function for POLICY iteration = {}".format(value_func))
        draw_value_func_heatmap(value_func, map_size)

        # c
        optimal_policy_converted = convert_values_to_optimal_policy(env, value_func)
        print("\nf. Convert optimal values to optimal policy")
        draw_policy_console(optimal_policy_converted, action_names, map_size)
        print(optimal_policy_converted == optimal_policy)
        # e
        total_reward = 0.
        iters = 1000
        for _ in range(iters):
            reward, num_steps = execute_optimal_policy_cummulative(env, optimal_policy, False)
            total_reward += reward
        print("g. Total reward = {}, average CUMMULATIVE DISCOUNTED reward after {} iters = {}".
              format(total_reward, iters, total_reward / iters))

        print("==============================================================================================\n\n\n")


def question_2_c():
    for env_name, map_size in [('Deterministic-4x4-neg-reward-FrozenLake-v0', 4)]:
        env = gym.make(env_name)
        print("\n\n\n==============================================================================================\n")
        print("Processing env {}".format(env_name))

        # a
        t1 = time.time()
        optimal_policy, value_func, count = rl.value_iteration(env, gamma=0.9)
        print("\n\n\na. Time taken for VALUE iteration for question 2.c is {} seconds".format(time.time() - t1))
        print("a. Total iterations is {}".format(count))
        print("\n\nOptimal policy for VALUE iteration is: ")
        draw_policy_console(optimal_policy, action_names, map_size)

        print("\nc. Value function for POLICY iteration = {}".format(value_func))
        draw_value_func_heatmap(value_func, map_size)

        # c
        optimal_policy_converted = convert_values_to_optimal_policy(env, value_func)
        print("\nf. Convert optimal values to optimal policy")
        draw_policy_console(optimal_policy_converted, action_names, map_size)


def main():
    # question_2_a()
    question_2_b()
    # question_2_c()


if __name__ == '__main__':
    main()
