# Bismillahi-r-Rahmani-r-Rahim

# Learn distributions for probabilistic semantics using expectation maximisation

import numpy
from numpy.random import RandomState
from itertools import product
from copy import deepcopy
import operator
import math
from math import log
import scipy.optimize

#random.seed(0)

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
        self.max_its = 100
        self.random = RandomState(1)

    def learn(self, input_theories):
        self.initialise(input_theories)

        x0 = self.get_flat()
        def objective(x):
            theta_old = self.theta
            p_h_old = self.p_h
            self.set_from_flat(x)
            log_probs = [log(self.prob(t)) for t in input_theories]
            self.theta = theta_old
            self.p_h = p_h_old
            obj = -sum(log_probs)
            #print "Objective %f, called with %r" % (obj, x)
            return obj

        #x, f, d = scipy.optimize.fmin_l_bfgs_b(objective, x0, approx_grad=True, factr=10, pgtol=1e-20, iprint=0, epsilon=1e-3, disp=1)
        bounds = [(0.0, 1.0)]*x0.size
        #x, f, d = scipy.optimize.fmin_l_bfgs_b(objective, x0, approx_grad=True, bounds=bounds, disp=1, pgtol=1e-10, factr=10)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(objective, x0, approx_grad=True, bounds=bounds, disp=1)
        self.set_from_flat(x)
        
    def set_from_flat(self, flat):
        theta_new = {k: numpy.zeros(v.shape) for k,v in self.theta.iteritems()}
        i = 0
        for k, v in sorted(theta_new.items()):
            v = theta_new[k]
            #print "Key", k, "Size", v.size
            v.flat = flat[i: i+v.size]
            i += v.size
        self.p_h = flat[i: i+self.p_h.size]
        self.theta = theta_new
        self.normalise()

    def get_flat(self):
        #print "Sorted keys", [x[0] for x in sorted(self.theta.items())]
        return numpy.concatenate([x.flat for k, x in sorted(self.theta.items())] + [self.p_h])

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
            assert numpy.all((v >= -1e-3) & (v <= 1.0 + 1e-3))

    def initialise(self, theories):
        self.theta = {}
        for theory in theories:
            for sentence in theory:
                truth, f, value = sentence
                if truth != 'truth':
                    raise ValueError('Expected truth assertion')
                self.initialise_sub(f)
            self.p_h = self.random.rand(self.hidden_dims)
        self.normalise()
            
    def initialise_sub(self, f):
        if f[0] == 'w':
            assert f[1] in words
            self.theta[f] = self.random.rand(self.dims[f[1]], self.hidden_dims)
        elif f[0] == 'f':
            assert f[1] in functions
            f2, f3 = f[2][1], f[3][1]
            self.theta[(f[0], f[1], f2, f3)] = self.random.rand(self.dims[f[1]], self.dims[f2], self.dims[f3], self.hidden_dims)
            self.initialise_sub(f[2])
            self.initialise_sub(f[3])
        else:
            raise ValueError('Should be "w" or "f"')

    def prob_all(self, theories):
        log_probs = [log(self.prob(t)) for t in theories]
        log_prob = sum(log_probs)
        return math.e**(log_prob)

    def prob(self, theory):
        #print "Theta", self.theta
        total_prob = 0.0
        for h in range(self.hidden_dims):
            h_prob = 0.0
            for theory_values, subs in self.substitute_values(theory):
                values_prob = reduce(operator.mul,
                                     self.subs_probs(subs, h),
                                     1.0)
                h_prob += values_prob
            assert h_prob < 1.0 + 1e-5
            total_prob += self.p_h[h]*h_prob
        assert total_prob >= 0.0 and total_prob <= 1.0
        return total_prob

    def ascend(self, theory):
        #print "Theta", self.theta
        #print "h:", self.p_h
        h_probs = []
        new_theta = {k:numpy.copy(v) for k,v in self.theta.iteritems()}
        old_prob = new_prob = 0.0
        #while new_prob < old_prob:
        #self.step *= 1.2
        #print "New step: ", self.step
        for h in range(self.hidden_dims):
            h_prob = 0.0
            for theory_values, subs in self.substitute_values(theory):
                prob_logs = [log(x) for x in self.subs_probs(subs, h)]
                values_prob_log = sum(prob_logs)
                for p_log, (key, value) in zip(prob_logs, subs):
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

        self.theta = new_theta
        self.normalise()
        #print "Theta", self.theta
        #print "h:", self.p_h

        new_prob = self.prob(theory)
        print "Probabilities:", new_prob, old_prob
        return new_prob > old_prob

            # if new_prob < old_prob:
            #     self.theta = old_theta
            #     self.p_h = old_h
            #     self.step *= 0.5
            #     print "Adjusting step downwards to %f" % self.step

            
    def subs_probs(self, subs, h):
        for key, value in subs:
            if key[0] == 'w':
                v = self.theta[key][value, h]
                assert v >= 0.0 and v <= 1.0
                yield v
            else:
                v = self.theta[key[:4]][value, key[4], key[5], h]
                assert v >= 0.0 and v <= 1.0
                yield v

    def substitute_values(self, theory, subs = ()):
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

    def substitute_expression_values(self, expression, subs = ()):
        if expression[0] == 'w':
            key = expression
            value = find(subs, key)
            if value is not None:
                yield expression + (value,), subs
            else:
                for i in range(self.dims[expression[1]]):
                    subs_copy = subs + ((key, i),)
                    #print "Subs1", subs_copy
                    yield expression + (i,), subs_copy
        else:
            for new_expression1, new_subs1 in self.substitute_expression_values(expression[2], subs):
                for new_expression2, new_subs2 in self.substitute_expression_values(expression[3], new_subs1):
                    key = (expression[0], expression[1], new_expression1[1], new_expression2[1], new_expression1[-1], new_expression2[-1])
                    value = find(new_subs2, key)
                    if value is not None:
                        yield expression[:2] + (new_expression1, new_expression2, value), new_subs2
                    else:
                        for i in range(self.dims[expression[1]]):
                            subs_copy = new_subs2 + ((key, i),)
                            #print "Subs2", subs_copy
                            yield expression[:2] + (new_expression1, new_expression2, i), subs_copy
                        
def find(to_search, key):
    for k, v in to_search:
        if k == key:
            return v
