# Bismillahi-r-Rahmani-r-Rahim

# Learn distributions for probabilistic semantics using expectation maximisation

import numpy
from numpy.random import rand
from itertools import product
from copy import deepcopy

det = 'det'
noun = 'noun'
verb = 'verb'
words = [det, noun, verb]

s = 's'
np = 'np'
vp = 'vp'
functions = [s, np, vp]

class Learner(object):
    def __init__(self):
        self.dims = {x: 2 for x in words + functions}
        self.hidden_dims = 2
        self.theta = {}
        self.p_h = numpy.array([0.5]*self.hidden_dims)

    def initialise(self, theory):
        for sentence in theory:
            truth, f, value = sentence
            if truth != 'truth':
                raise ValueError('Expected truth assertion')
            self.initialise_sub(f)
        self.p_h = rand(self.hidden_dims)

    def initialise_sub(self, f):
        if f[0] == 'w':
            assert f[1] in words
            self.theta[f] = rand(self.dims[f[1]], self.hidden_dims)
        elif f[0] == 'f':
            assert f[1] in functions
            f2, f3 = f[2][1], f[3][1]
            self.theta[(f[0], f[1], f2, f3)] = rand(self.dims[f[1]], self.dims[f2], self.dims[f3], self.hidden_dims)
            self.initialise_sub(f[2])
            self.initialise_sub(f[3])
        else:
            raise ValueError('Should be "w" or "f"')

    def prob(self, theory):
        print self.theta
        total_prob = 0.0
        for h in range(self.hidden_dims):
            prob = 1.0
            for theory_values, subs in self.substitute_values(theory, {}):
                for key, value in subs.iteritems():
                    if key[0] == 'w':
                        p = self.theta[key][h, value]
                        prob *= p
                    else:
                        p = self.theta[key[:4]][h, key[4], key[5], value]
                        prob *= p
            total_prob += self.p_h[h]*prob
        return total_prob
            
        
    def substitute_values(self, theory, subs):
        """
        Returns an iterable of all possible value assignments for a theory
        """
        if len(theory) == 0:
            yield (), subs
            return
        truth, sentence, value = theory[0]
        for expression, new_subs in self.substitute_expression_values(sentence, subs):
            if expression[-1] != value:
                continue
            for remainder, remainder_subs in self.substitute_values(theory[1:], new_subs):
                yield (expression,) + remainder, remainder_subs

    def substitute_expression_values(self, expression, subs):
        if expression[0] == 'w':
            key = expression
            if key in subs:
                yield expression + (subs[key],), subs
            else:
                for i in range(self.dims[expression[1]]):
                    subs_copy = deepcopy(subs)
                    subs_copy[key] = i
                    yield expression + (i,), subs_copy
        else:
            for new_expression1, new_subs1 in self.substitute_expression_values(expression[2], subs):
                for new_expression2, new_subs2 in self.substitute_expression_values(expression[3], new_subs1):
                    key = (expression[0], expression[1], new_expression1[1], new_expression2[1], new_expression1[-1], new_expression2[-1])
                    if key in new_subs2:
                        yield expression[:2] + (new_expression1, new_expression2, new_subs2[key]), new_subs2
                    else:
                        for i in range(self.dims[expression[1]]):
                            subs_copy = deepcopy(new_subs2)
                            subs_copy[key] = i
                            yield expression[:2] + (new_expression1, new_expression2, i), subs_copy
                        
                

