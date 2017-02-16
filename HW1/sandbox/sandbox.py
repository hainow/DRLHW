# a = np.asarray(range(16)).reshape(4,4)
# x = np.zeros((1, 4))
# y = np.zeros((6, 1))
# print a
# print x, y
#
# a = np.append(a, x, axis=0)
# a = np.append(x, a, axis=0)
# print a
# a = np.append(y, a, axis=1)
# a = np.append(a, y, axis=1)
# print a
#
#
#
import gym
import HW1.question2 as q2
#
env = gym.make('Stochastic-4x4-FrozenLake-v0')
# env = gym.make('Deterministic-4x4-FrozenLake-v0')
# print env.isd
q2.print_transition(env)
# transitions = env.P[0][1]
# s = [t[0] for t in transitions]
# print s

# env.render()



# def convert_tuple_to_state_number(tuple):
#     """ tuple = (a, b, c, d)"""
#     return tuple[0] * 6**3 + tuple[1] * 6**2 + tuple[2] * 6 + tuple[3]
#
# def convert_tuple_to_state_number(a, b, c, d):
#     """ tuple = (a, b, c, d)"""
#     return a * 6**3 + b * 6**2 + c * 6 + d
#
# li = []
# for a in range(3):
#     for b in range(6):
#         for c in range(6):
#             for d in range(6):
#                 li.append((a, b, c, d))
# print len(li)
# print len(li) == 3 * 6**3





# def build_increment(p1, p2, p3):
#     """
#     For each action, we have 8 possible next states, with increment of items
#     (0, 0, 0), (0, 0, 1), ... , (1, 1, 1) with probablility (1-p1)(1-p2)(1-p3), (1-p1)(1-p2)p3, ...., p1*p2*p3
#     :return: a dictionary mapping the 2 lists above
#     """
#     l = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
#     pa = {0: (1 - p1), 1: p1}
#     pb = {0: (1 - p2), 1: p2}
#     pc = {0: (1 - p3), 1: p3}
#     print pa, pb, pc
#     x = dict()
#
#     for tuple in l:
#         x[tuple] = pa[tuple[0]] * pb[tuple[1]] * pc[tuple[2]]
#
#     return x
#
# x = build_increment(.1, .9, .1)
#
# for k in x:
#     print k, x[k]