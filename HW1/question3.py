import gym
import time

import deeprl_hw1.queue_envs as queue_envs
import deeprl_hw1.rl as rl


def print_transition(env):
    print("\nTransition policy of this env:")
    for s in env.states:
        print('State {} ->'.format(s))
        for a in range(env.nA):
            print ('\taction {}'.format(a))
            for p, s1, s2, s3, s4, r, done in env.P[s][a]:
                print ('\t\t ===> p={} s_next=({} {} {} {}) r={} done={}'.
                       format(p, s1, s2, s3, s4, r, done))




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
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 1
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal or num_steps > max_steps:
            break

        time.sleep(.5)

    return total_reward





def main():
    env = gym.make('Queue-1-v0')
    print_transition(env)

    # env.render()

    iters = 30
    total_reward = run_random_policy(env, max_steps=iters)
    print("Total reward = {} after {} iters, average = {}".format(total_reward, iters, total_reward / float(iters)))


if __name__ == "__main__":
    main()