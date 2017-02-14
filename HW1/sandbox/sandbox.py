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

# env = gym.make('Stochastic-4x4-FrozenLake-v0')
env = gym.make('Deterministic-4x4-FrozenLake-v0')
print env.isd
q2.print_transition(env)
env.render()



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