# Bismillahi-r-Rahmani-r-Rahim

# Learn distributions for probabilistic semantics using expectation maximisation

import numpy
from numpy import random
from itertools import product
from copy import deepcopy
import operator
from random import Random
import math
from math import log

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
        self.initial_step = 0.5
        self.step = self.initial_step

    def learn(self, theories):
        self.initialise(theories)
        self.step = self.initial_step
        r = Random(1)
        shuffled = deepcopy(theories)
        r.shuffle(shuffled)
        log_probs = [log(self.prob(t)) for t in theories]
        old_log_prob = reduce(operator.add, log_probs, 0.0)
        for i in range(10):
            print "Learn iteration:", i, "Probability:", old_log_prob
            for t in shuffled:
                self.ascend(t)
            log_probs = [log(self.prob(t)) for t in theories]
            log_prob = reduce(operator.add, log_probs, 0.0)
            assert log_prob > old_log_prob
            #if log_prob/old_log_prob > 0.999:
            #    break
            old_log_prob = log_prob

    def normalise(self):
        self.p_h /= numpy.sum(self.p_h)
        for h in range(self.hidden_dims):
            for key, v in self.theta.iteritems():
                if key[0] == 'w':
                    v[:, h] /= numpy.sum(v[:,h])
                else:
                    for i, j in product(*[range(x) for x in v.shape[1:3]]):
                        v[:, i, j, h] /= numpy.sum(v[:, i, j, h])

        for key, v in self.theta.iteritems():
            assert numpy.all((v >= 0.0) & (v <= 1.0))

    def initialise(self, theories):
        self.theta = {}
        for theory in theories:
            for sentence in theory:
                truth, f, value = sentence
                if truth != 'truth':
                    raise ValueError('Expected truth assertion')
                self.initialise_sub(f)
            self.p_h = random.rand(self.hidden_dims)
        self.normalise()
            
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
        #print "Theta", self.theta
        total_prob = 0.0
        for h in range(self.hidden_dims):
            h_prob = 0.0
            for theory_values, subs in self.substitute_values(theory, {}):
                values_prob = reduce(operator.mul,
                                     self.subs_probs(subs, h),
                                     1.0)
                h_prob += values_prob
            assert h_prob <= 1.0
            total_prob += self.p_h[h]*h_prob
        assert total_prob >= 0.0 and total_prob <= 1.0
        return total_prob

    def ascend(self, theory):
        print "Theta", self.theta
        print "h:", self.p_h
        h_probs = []
        new_theta = {k:numpy.copy(v) for k,v in self.theta.iteritems()}
        old_prob = new_prob = 0.0
        #while new_prob < old_prob:
        #self.step *= 1.2
        print "New step: ", self.step
        for h in range(self.hidden_dims):
            h_prob = 0.0
            for theory_values, subs in self.substitute_values(theory, {}):
                prob_logs = [log(x) for x in self.subs_probs(subs, h)]
                values_prob_log = reduce(operator.add, prob_logs, 0.0)
                for p_log, (key, value) in zip(prob_logs, subs.items()):
                    p_exclusive = math.e**(values_prob_log - p_log)
                    delta = self.step*self.p_h[h]*p_exclusive
                    if key[0] == 'w':
                        new_theta[key][value, h] += delta
                    else:
                        new_theta[key[:4]][value, key[4], key[5], h] += delta
                values_prob = math.e**values_prob_log
                h_prob += values_prob
            h_probs.append(h_prob)
        old_prob = 0.0
        for i, h_prob in enumerate(h_probs):
            delta = self.step*h_prob
            old_prob += self.p_h[i]*h_prob
            self.p_h[i] += delta
        # old_prob2 = self.prob(theory)
        # assert abs(old_prob2 - old_prob) <= 1e-5

        old_theta = self.theta
        old_h = self.p_h
        self.theta = new_theta
        self.normalise()
        print "Theta", self.theta
        print "h:", self.p_h

        new_prob = self.prob(theory)
        print "Probabilities:", new_prob, old_prob
        assert new_prob > old_prob

            # if new_prob < old_prob:
            #     self.theta = old_theta
            #     self.p_h = old_h
            #     self.step *= 0.5
            #     print "Adjusting step downwards to %f" % self.step

            
    def subs_probs(self, subs, h):
        for key, value in subs.iteritems():
            if key[0] == 'w':
                v = self.theta[key][value, h]
                assert v >= 0.0 and v <= 1.0
                yield v
            else:
                v = self.theta[key[:4]][value, key[4], key[5], h]
                assert v >= 0.0 and v <= 1.0
                yield v

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
                        
                

