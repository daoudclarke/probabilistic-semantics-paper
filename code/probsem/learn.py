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
        
    def substitute_values(self, theory):
        """
        Returns an iterable of all possible value assignments for a theory
        """
        
    def substitute_expression_values(self, expression, subs):
        if expression[0] == 'w':
            key = (expression[1], expression[2])
            expression_copy = deepcopy(expression)
            expression_copy.append(None)
            if key in subs:
                expression_copy[-1] = subs[key]
                yield expression_copy, subs
            else:
                for i in range(self.dims[expression[1]]):
                    subs_copy = deepcopy(subs)
                    subs_copy[key] = i
                    expression_copy[-1] = i
                    yield expression_copy, subs_copy
        else:
            for new_expression1, new_subs1 in self.substitute_expression_values(expression[2], subs):
                for new_expression2, new_subs2 in self.substitute_expression_values(expression[3], new_subs1):
                    key = (expression[1], new_expression1[-1], new_expression2[-1])
                    expression_copy = deepcopy(expression)
                    expression_copy.append(None)
                    expression_copy[2] = new_expression1
                    expression_copy[3] = new_expression2
                    if key in new_subs2:
                        expression_copy[-1] = new_subs2[key]
                        yield expression_copy, new_subs2
                    else:
                        for i in range(self.dims[expression[1]]):
                            subs_copy = deepcopy(new_subs2)
                            subs_copy[key] = i
                            expression_copy[-1] = i
                            yield expression_copy, subs_copy
                        
                

