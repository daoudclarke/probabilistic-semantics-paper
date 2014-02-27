from probsem.learn import Learner, words, functions
import pytest
import logging
import sys
import numpy
import cPickle as pickle
from random import Random

@pytest.fixture
def sentence():
    return make_sentence('some cats like all dogs')

def make_sentence(text):
    words = text.split()
    s = 's'
    det = 'det'
    noun = 'noun'
    verb = 'verb'
    np = 'np'
    vp = 'vp'
    def f(p, l1, l2):
        return ('f', p, l1, l2)
    def w(w, p):
        return ('w', w, p)
    return f(s, f(np, w(det, words[0]), w(noun, words[1])),
             f(vp, w(verb, words[2]), f(np, w(det, words[3]), w(noun, words[4]))))

@pytest.fixture
def learner():
    s = sentence()
    theory = [truth(s, True)]
    l = Learner()
    l.initialise([theory])
    return l

def truth(l, v):
    return ['truth', l, v]

@pytest.fixture
def train_sentences():
    train_sentences = [('some cats like all dogs', 'some animals like all dogs'),
                       ('no animals like all dogs', 'no cats like all dogs'),
                       ('some dogs like all dogs', 'some animals like all dogs'),
                       ('no animals like all dogs', 'no dogs like all dogs'),
                       ('some men like all dogs', 'some people like all dogs')]
    return [(make_sentence(t), make_sentence(h)) for t,h in train_sentences]

@pytest.fixture
def train():
    train = train_sentences()
    train_data = [(truth(text, True), truth(hypothesis, True))
                  for text, hypothesis in train]
    train_data += [(truth(text, False),)
                   for text, hypothesis in train]
    return train_data

# @pytest.fixture
# def train():
#     train = train_sentences()
#     train_data = [(truth(text, True), truth(hypothesis, True))
#                   for text, hypothesis in train]
#     train_data += [(truth(text, False), truth(hypothesis, True))
#                    for text, hypothesis in train]
#     train_data += [(truth(text, False), truth(hypothesis, False))
#                    for text, hypothesis in train]
#     return train_data

@pytest.fixture
def test():
    test_sentences = [('no people like all dogs','no men like all dogs', True),
                      ('no men like all dogs','no people like all dogs', False)]
    return [(make_sentence(t), make_sentence(h), v) for t, h, v in test_sentences]
    

# ================================ TESTS ======================================

#@pytest.mark.xfail
def test_learn_simple(train, train_sentences):
    learner = Learner()
    # learner.hidden_dims = 2
    # learner.dims['noun'] = 3
    # learner.dims['det'] = 3
    # learner.dims['verb'] = 3
    # learner.dims['np'] = 3
    # learner.dims['vp'] = 3

    learner.learn(train)
    print learner.theta

    for text, hypothesis in train_sentences:
        p_t = learner.prob([truth(text, True)])
        p_th = learner.prob([truth(text, True), truth(hypothesis, True)])
        entailment = p_th/p_t
        print "Text: ", text
        print "Hypothesis: ", hypothesis
        assert entailment > 0.85

#@pytest.mark.xfail
def test_learn_finds_local_maximum(train, train_sentences):
    learner = Learner()
    data = [train[0], train[5]]
    learner.learn(data)
    print learner.theta

    prob = learner.prob_all(data)
    r = Random(1)
    for i in range(10):
        key = r.choice(learner.theta.keys())
        delta = r.choice([0.1, -0.1])
        index = r.randint(0, len(learner.theta[key].flat) - 1)
        learner.theta[key].flat[index] += delta
        learner.normalise()
        new_prob = learner.prob_all(data)
        print "Check %d, key %s, index %d" % (i, key, index)
        assert new_prob <= prob

def test_before_learn(train, test):
    learner = Learner()
    learner.initialise(train)

    allTrue = True
    for text, hypothesis, expected in test:
        p_t = learner.prob([truth(text, True)])
        p_th = learner.prob([truth(text, True), truth(hypothesis, True)])
        entailment = p_th/p_t
        assert entailment <= 1.0 and entailment >= 0.0
        allTrue = allTrue and ((entailment > 0.9) == expected)
    assert not allTrue

def test_initialise(sentence, learner):
    data = [truth(sentence, True)]
    print data
    print "Theta: ", learner.theta

    assert len(learner.theta[('w', 'det', 'some')]) > 0
    assert len(learner.theta[('f', 's', 'np', 'vp')]) > 0

    # Check normalisation
    a = learner.theta[('w', 'det', 'some')]
    assert abs(numpy.sum(a[:,1]) - 1.0) <= 1e-5


def test_prob_consistency(sentence, learner):
    data1 = [truth(sentence, True)]
    data2 = [truth(sentence, True), truth(sentence, True)]
    
    prob1 = learner.prob(data1)
    prob2 = learner.prob(data2)

    print data1

    assert prob1 >= 0.0
    assert prob1 <= 1.0
    assert prob1 == prob2

def test_prob_entailment_not_greater_than_one(test):
    learner = Learner()
    pickle_file = open('data/theta.pickle')
    learner.p_h, learner.theta = pickle.load(pickle_file)
    learner.normalise()
    print learner.theta
    text, hypothesis, expected = test[1]
    p_t = learner.prob([truth(text, True)])
    print "========================================================"
    p_th = learner.prob([truth(text, True), truth(hypothesis, True)])
    assert p_t >= 0 and p_t <= 1
    assert p_th >= 0 and p_th <= 1
    assert p_th <= p_t

def test_prob_bounds(sentence):
    data = [truth(sentence, True)]
    learner = Learner()
    learner.hidden_dims = 2
    learner.dims['noun'] = 3
    learner.dims['det'] = 4
    learner.dims['np'] = 5
    learner.initialise([data])
    prob = learner.prob(data)

def test_prob_contradiction(sentence, learner):
    data = [truth(sentence, True), truth(sentence, False)]
    print data
    prob = learner.prob(data)
    assert prob == 0.0

def test_prob_sum_contradictions(sentence, learner):
    data1 = [truth(sentence, True)]
    data2 = [truth(sentence, False)]
    prob1 = learner.prob(data1)
    prob2 = learner.prob(data2)
    assert abs(prob1 + prob2 - 1.0) <= 1e-5

def test_substitute_values(sentence, learner):
    data = (truth(sentence, True),)

    results = list(learner.substitute_values(data, 0))
    print "First ten expressions:"
    for x, subs, prob in results[:10]:
        print x, subs
        assert len(subs) > 0

def test_substitute_true_and_false(sentence, learner):
    data1 = (truth(sentence, True),)
    data2 = (truth(sentence, False),)

    results1 = list(learner.substitute_values(data1, 0))
    results2 = list(learner.substitute_values(data2, 0))

    for x, subs, prob in results1[:10]:
        print x, subs
        assert len(subs) > 0
    
    s1 = set(x[0] for x in results1)
    s2 = set(x[0] for x in results2)

    assert len(s1 & s2) == 0
    assert len(results1) == len(results2)
        
def test_substitute_values_with_duplicates(sentence, learner):
    data1 = (truth(sentence, True),)
    data2 = (truth(sentence, True), truth(sentence, True))
    
    results1 = list(learner.substitute_values(data1, 0))
    results2 = list(learner.substitute_values(data2, 0))
    assert len(results1) == len(results2)

def test_substitute_values_contradiction(sentence, learner):
    data = (truth(sentence, True), truth(sentence, False))
    
    results = list(learner.substitute_values(data, 0))
    assert len(results) == 0
    
def test_substitute_expression_word(learner):
    expression = ('w', 'det', 'some')
    values = list(learner.substitute_expression_values(expression, 0))

    assert len(values) == 2
    for e, subs, prob in values:
        print e, subs
        assert len(subs) == 1
        assert e[:-1] == expression
        assert e[-1] in [0,1]
        
def test_substitute_expression_recursion(learner):
    expression = ('f', 'np', ('w', 'det', 'some'), ('w', 'noun', 'cats'))
    values = list(learner.substitute_expression_values(expression, 0))
    assert len(values) == 8
    for e, subs, prob in values:
        print e, subs
        assert len(subs) == 3
        assert e[:2] == ('f', 'np')
        assert e[-1] in [0,1]
        assert e[2][-1] in [0,1]
        assert e[3][-1] in [0,1]

@pytest.mark.xfail
def test_substitute_expression_repeated_values(learner):
    expression = ('f', 'np', ('w', 'det', 'some'), ('w', 'det', 'some'))
    values = list(learner.substitute_expression_values(expression, 0))
    assert len(values) == 4

def test_gradient(learner, sentence):
    data = (truth(sentence, False),)

    prob_before = learner.prob(data)
    gradient = learner.ascend(data)
    prob_after = learner.prob(data)
    
    assert prob_after > prob_before
