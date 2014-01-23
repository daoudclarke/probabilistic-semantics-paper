# Bismillahi-r-Rahmani-r-Rahim

# Learn distributions for probabilistic semantics using expectation maximisation

from numpy.random import rand

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

    def initialise(self, theory):
        for sentence in theory:
            truth, f, value = sentence
            if truth != 'truth':
                raise ValueError('Expected truth assertion')
            self.initialise_sub(f)

    def initialise_sub(self, f):
        if f[0] == 'w':
            assert f[1] in words
            self.theta[tuple(f[1:])] = rand(self.dims[f[1]], self.hidden_dims)
        elif f[0] == 'f':
            assert f[1] in functions
            f2, f3 = f[2][1], f[3][1]
            self.theta[(f[1], f2, f3)] = rand(self.dims[f[1]], self.dims[f2], self.dims[f3], self.hidden_dims)
            self.initialise_sub(f[2])
            self.initialise_sub(f[3])
        else:
            raise ValueError('Should be "w" or "f"')

    def prob(self, theory):
        for h in range(self.hidden_dims):
            cached = {}
            for sentence in theory:
                truth, f, value = sentence
                if truth != 'truth':
                    raise ValueError('Expected truth assertion')
                self.prob_sub(f, h, cached)
        

    def prob_sub(self, f, h):
        
