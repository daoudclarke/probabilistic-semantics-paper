# Bismillahi-r-Rahmani-r-Rahim

# Learn distributions for probabilistic semantics using expectation maximisation

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
        self.words = {}
        self.functions = {}

    def initialise(self, theory):
        for sentence in theory:
            truth, f, value = sentence
            if truth != 'truth':
                raise ValueError('Expected truth assertion')
            self.initialise_sub(f)

    def initialise_sub(self, f):
        if f[0] == 'w':
            assert f[1] in words
            self.words[tuple(f[1:])] = rand(self.dims[f[1]], self.hidden_dims)
        elif f[0] == 'f':
            assert f[1] in functions
            f2, f3 = f[2][1], f[3][1]
            self.functions[(f[1], f2, f3)] = rand(self.dims[f[1]], self.dims[f2], self.dims[f3], self.hidden_dims)
            self.initialise_sub(f[2])
            self.initialise_sub(f[3])
        else:
            raise ValueError('Should be "w" or "f"')

    def prob(self, theory):
        for h in range(self.hidden_dims):
            # word_indexes = [range(self.dims[x[0]]) for x in self.words.keys()]
            # function_indexes = [range(self.dims[x[0]]) for x in self.functions.keys()]
            # for x in product(*(word_indexes + function_indexes)):
            #     for values, index in zip(self.words.items(), x[:len(self.words)]):
            #         print index, values[1][h, index]

            #     for values, index in zip(self.functions.items(), x[len(self.words):]):
            #         print index, values[1][h, index]
            
            cached = {}
        
            prob = 1.0
            for sentence in theory:
                truth, f, value = sentence
                if truth != 'truth':
                    raise ValueError('Expected truth assertion')
                sentence_prob = self.prob_sub(f, value, cached)
                prob *= sentence_prob
            return prob
        
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
            key = (expression[1], expression[2])
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
                    key = (expression[1], new_expression1[-1], new_expression2[-1])
                    if key in new_subs2:
                        yield expression[:2] + (new_expression1, new_expression2, new_subs2[key]), new_subs2
                    else:
                        for i in range(self.dims[expression[1]]):
                            subs_copy = deepcopy(new_subs2)
                            subs_copy[key] = i
                            yield expression[:2] + (new_expression1, new_expression2, i), subs_copy
                        
                

