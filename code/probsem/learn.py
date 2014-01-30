# Bismillahi-r-Rahmani-r-Rahim

# Learn distributions for probabilistic semantics using expectation maximisation

import numpy
from numpy import random
from itertools import product
from copy import deepcopy
import operator

random.seed(1)

import logging

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
        self.p_h = random.rand(self.hidden_dims)
        self.normalise()

    def normalise(self):
        self.p_h /= numpy.sum(self.p_h)

    def initialise_sub(self, f):
        if f[0] == 'w':
            assert f[1] in words
            self.theta[f] = random.rand(self.dims[f[1]], self.hidden_dims)
        elif f[0] == 'f':
            assert f[1] in functions
            f2, f3 = f[2][1], f[3][1]
            self.theta[(f[0], f[1], f2, f3)] = random.rand(self.dims[f[1]], self.dims[f2], self.dims[f3], self.hidden_dims)
            self.initialise_sub(f[2])
            self.initialise_sub(f[3])
        else:
            raise ValueError('Should be "w" or "f"')

    def prob(self, theory):
        print "Theta", self.theta
        total_prob = 0.0
        for h in range(self.hidden_dims):
            h_prob = 0.0
            for theory_values, subs in self.substitute_values(theory, {}):
                values_prob = reduce(operator.mul,
                                     self.subs_probs(subs, h),
                                     1.0)
                h_prob += values_prob
            total_prob += self.p_h[h]*h_prob
        return total_prob

    def ascend(self, theory, step):
        print "Theta", self.theta
        h_probs = []
        for h in range(self.hidden_dims):
            h_prob = 0.0
            for theory_values, subs in self.substitute_values(theory, {}):
                probs = list(self.subs_probs(subs, h))
                values_prob = reduce(operator.mul, probs, 1.0)
                for p, (key, value) in zip(probs, subs.items()):
                    delta = step*self.p_h[h]*values_prob/p
                    if key[0] == 'w':
                        self.theta[key][h, value] += delta
                    else:
                        self.theta[key[:4]][h, key[4], key[5], value] += delta
                    
                h_prob += values_prob
            h_probs.append(h_prob)
        for i, h_prob in enumerate(h_probs):
            delta = step*h_prob
            self.p_h[i] += delta
        self.normalise()

            
    def subs_probs(self, subs, h):
        for key, value in subs.iteritems():
            if key[0] == 'w':
                yield self.theta[key][h, value]
            else:
                yield self.theta[key[:4]][h, key[4], key[5], value]

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
                        
                

