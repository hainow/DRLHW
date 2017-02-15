# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

import numpy as np

from gym import Env, spaces
from gym.envs.registration import register
from six import StringIO, b
from gym import utils  # for coloring

def categorical_sample(prob_n):  # taken from discrete.py
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    # 4 actions
    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3


    def __init__(self, p1, p2, p3):

        def build_increment(p1, p2, p3):
            """
            For each action, we have 8 possible next states, with increment of items
            (0, 0, 0), (0, 0, 1), ... , (1, 1, 1) with probablility (1-p1)(1-p2)(1-p3), (1-p1)(1-p2)p3, ...., p1*p2*p3
            :return: a dictionary mapping the 2 lists above
            """
            l = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
            pa = {0: (1 - p1), 1: p1}
            pb = {0: (1 - p2), 1: p2}
            pc = {0: (1 - p3), 1: p3}
            x = dict()

            for tuple in l:
                x[tuple] = pa[tuple[0]] * pb[tuple[1]] * pc[tuple[2]]

            return x

        def upper_bound(array, upper=5):
            array[array>5] = 5
            return array


        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 3 * 6 * 6 * 6
        self.nA = 4
        self.last_action = None  # for rendering
        isd = np.array([1] + [0] * (self.nS-1)) #TODO: ask about whether we need to sample how many items pre-existed?



        # self.P = dict
        self.states = [(a, b, c, d) for a in range(3) for b in range(6) for c in range(6) for d in range(6)]
        self.P = {s: {a: [] for a in range(self.nA)} for s in self.states}
        increments = build_increment(p1, p2, p3)

        for a in range(3):
            for b in range(6):
                for c in range(6):
                    for d in range(6):
                        current_state = (a, b, c, d)
                        for action in range(4):
                            li = self.P[current_state][action]  # currently empty
                            interim_state = None
                            if action <3:
                                interim_state = np.asarray((action, b, c, d))
                                # serve now ...
                                interim_state[action + 1] -= 1

                                # get reward
                                reward = 0
                                if interim_state[action + 1] < 0:
                                    interim_state[action + 1] = 0
                                else:
                                    reward = 1

                                # update env.P[s][a]
                                for key in increments:
                                    next_state = upper_bound(interim_state + np.array((0,) + key))
                                    prob = increments[key]
                                    li.append( (prob,) + (tuple(next_state), ) + (reward, False) )

                            else:  # action == 3: stay and serve current queue
                                interim_state = np.asarray((a, b, c, d))  # a unchanged
                                # serve now ...
                                interim_state[a + 1] -= 1

                                # get reward
                                reward = 0
                                if interim_state[a + 1] < 0:
                                    interim_state[a + 1] = 0
                                else:
                                    reward = 1

                                # update env.P[s][a]
                                for key in increments:
                                    next_state = upper_bound(interim_state + np.array((0,) + key))
                                    prob = increments[key]
                                    li.append( (prob,) + (tuple(next_state), ) + (reward, False) )

        # TODO: validate whether this is needed
        self._seed()
        self._reset()


    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        import random
        b = random.randint(0, 5)
        c = random.randint(0, 5)
        d = random.randint(0, 5)
        self.s = (0, b, c, d)

        return self.s

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions])
        p, next_state, r, d = transitions[i]
        debug = 'none'

        # update object attributes
        self.last_action = action
        self.s = next_state

        return (self.s, r, d, debug)

    def _render(self, mode='human', close=False):

        if close:
            return

        # decorate now
        s = np.array(self.s, dtype='c').tolist()
        if self.last_action is not None:
            if self.last_action < 3:
                s[self.last_action + 1] = utils.colorize(s[self.last_action + 1], "red", highlight=True)
            else:
                s[int(s[0]) + 1] = utils.colorize(s[int(s[0]) + 1], "red", highlight=True)
        else:
            s[1] = utils.colorize(s[1], "red", highlight=True)

        (a, b, c, d) = s
        # now output to console
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        if self.last_action is not None:
            outfile.write("\n\t\t\t   ACTION:  {} -->\n".
                          format(utils.colorize(["To Q1","To Q2","To Q3","Stay"][self.last_action],
                                                "green", highlight=False)))
        else:
            outfile.write("\n")

        outfile.write('\t\t----------------------------\n')
        outfile.write('\t\t[now]\t[Q1]\t[Q2]\t[Q3]\n')
        outfile.write('\t\t----------------------------\n')
        outfile.write('\t\t  {}\t {}\t {}\t {}\n'.format(a, b, c, d))

        outfile.write('\n\n')

        return outfile



    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        return np.random.seed(seed), np.random.random()

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        return self.P[state][action]

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
