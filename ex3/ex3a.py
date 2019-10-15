import numpy as np

#
# Q: Find p(blue | blond or red)
#
# +-----------------------------------------+
# |       | Black | Brunette | Red  | Blond |
# +-------+-------+----------+------+-------+
# | Brown |       |          | 0.04 |  0.01 |
# |  Blue |       |          |=0.03=|==0.16=|
# | Hazel |       |          | 0.02 |  0.02 |
# | Green |       |          | 0.02 |  0.03 |
# +-----------------------------------------+
# 
# Figure 1: Cells marked with `=` belong to the numerator of eq. 9. Cells with
#           probabilities given belong to the denominator in eq. 9.
#
# ```
#   p(c|r) = p(r,c) / p(r)                                                 (9)
# ```
#
# Here the condition ("blond or red") narrows the possible space to the two
# columnes so named. In this narrowed space the question becomes p(blue) which
# is then the marginal probability evaluated at `eye color = blue`.
# 
# This is connected to eq. 9 as we can reformulate the question as such:
# ```
#   p(eye color = blue | hair color = blond or red) = 
#       p(eye color = blue, hair color = blond or red) /
#       p(hair color = blond or red)
# ```
#
# Here `p(hair color = blond or red)` represents the column selection descibed
# in the previous paragraph and `p(eye color = blue, hair color = blond or red)`
# represents the narrowed marginal probability.
# 
# For completeness: (0.03+0.16) / (0.11 + 0.21) = 0.58
#



#
# === Question 2 ===
#
# Matrix form
#
# p('+, :(') | p('-, :(')
# p('+, :)') | p('-, :)')
# 
# which is equiv. to:
# 
# p('+|:(')p(':(') | p('-|:(')p(':(')
# p('+|:)')p(':)') | p('-|:)')p(':)')
#
# The below code was useful for debugging:
#
# p[':(|+'] = p['+|:('] * (p[':(']/p['+'])
# p[':)|+'] = p['+|:)'] * (p[':)']/p['+'])
# p[':(|-'] = p['-|:('] * (p[':(']/p['-'])
# p[':)|-'] = p['-|:)'] * (p[':)']/p['-'])
#
# p[':(|+,-'] = ((p['-|:('] / p['-']) *
#                (p['+|:('] / p['+']) *
#                (p[':(']   / 1     ))
#
# print('p[\':(|+\']   = {}'.format(p[':(|+']))
# print('p[\':(|-\']   = {}'.format(p[':(|-']))
# print('p[\':(|+,-\'] = {}'.format(p[':(|+,-']))
#
# M = np.asarray([[p['+|:(']*p[':('], p['-|:(']*p[':(']],
#                 [p['+|:)']*p[':)'], p['-|:)']*p[':)']]])
# print(M)
# print(np.sum(M, axis=0), np.sum(np.sum(M, axis=0)))
# print(np.sum(M, axis=1), np.sum(np.sum(M, axis=1)))
# print(np.sum(M))

p = {}

# Conditional
p['+|:('] = 0.99 # True positive
p['+|:)'] = 0.05 # False positive
p['-|:('] = 1 - p['+|:(']
p['-|:)'] = 1 - p['+|:)']

# Marginal
p[':('] = 0.001 # p sick
p[':)'] = 1 - p[':('] # p healthy
p['+'] = p['+|:(']*p[':('] + p['+|:)']*p[':)']
p['-'] = 1 - p['+']

def proba_is_sick(test_scores):
    '''
    The function call `proba_is_sick('+')` is equivalent to:
    ```
    p[':(|+'] = p['+|:('] * (p[':(']/p['+'])
    ```
    and `proba_is_sick('+-')` is equivalent to:
    ```
    p[':(|+,-'] = ((p['-|:('] / p['-']) *
                   (p['+|:('] / p['+']) *
                   (p[':(']   / 1     ))
    ```
    '''
    allowed_symbols = '-+'

    posterior = p[':(']
    for symbol in test_scores:
        assert(symbol in allowed_symbols)
        prior = posterior
        likelihood = p[f'{symbol}|:(']
        evidence = p[f'{symbol}']
        posterior = likelihood / evidence * prior

    return posterior

print(f'proba_is_sick(""): {proba_is_sick("")}\n'
      f'proba_is_sick("+"): {proba_is_sick("+")}\n'
      f'proba_is_sick("-"): {proba_is_sick("-")}\n'
      f'proba_is_sick("-+"): {proba_is_sick("-+")}\n'
      f'proba_is_sick("+-"): {proba_is_sick("+-")}')
