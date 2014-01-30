from probsem.learn import Learner, words, functions
import pytest
import logging
import sys
import numpy

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
def train():
    train_sentences = [('some cats like all dogs', 'some animals like all dogs'),
                       ('no animals like all dogs', 'no cats like all dogs'),
                       ('some dogs like all dogs', 'some animals like all dogs'),
                       ('no animals like all dogs', 'no dogs like all dogs'),
                       ('some men like all dogs', 'some people like all dogs')]
    train = [(make_sentence(t), make_sentence(h)) for t,h in train_sentences]
    train_data = [(truth(text, True), truth(hypothesis, True))
                  for text, hypothesis in train]
    train_data += [(truth(text, False), truth(hypothesis, True))
                   for text, hypothesis in train]
    train_data += [(truth(text, False), truth(hypothesis, False))
                   for text, hypothesis in train]
    return train_data

@pytest.fixture
def test():
    test_sentences = [('no people like all dogs','no men like all dogs', True),
                      ('no men like all dogs','no people like all dogs', False),
                      ('most people like all dogs','most men like all dogs', False)]
    return [(make_sentence(t), make_sentence(h), v) for t, h, v in test_sentences]
    

# ================================ TESTS ======================================

# @pytest.mark.xfail
# def test_learn(train, test):
#     learner = Learner()
#     learner.learn(train)

#     for text, hypothesis, expected in test:
#         p_t = learner.prob([truth(text, True)])
#         p_th = learner.prob([truth(text, True), truth(hypothesis, True)])
#         entailment = p_th/p_t
#         assert (entailment > 0.9) == expected

def test_before_learn(train, test):
    learner = Learner()
    learner.initialise(train)

    text, hypothesis, expected = test[0]
    p_t = learner.prob([truth(text, True)])
    p_th = learner.prob([truth(text, True), truth(hypothesis, True)])
    entailment = p_th/p_t
    assert (entailment > 0.9) != expected

def test_initialise(sentence, learner):
    data = [truth(sentence, True)]
    print data
    print "Theta: ", learner.theta

    assert len(learner.theta[('w', 'det', 'some')]) > 0
    assert len(learner.theta[('f', 's', 'np', 'vp')]) > 0

    # Check normalisation
    a = learner.theta[('w', 'det', 'some')]
    assert abs(numpy.sum(a[1,:]) - 1.0) <= 1e-5


def test_consistency(sentence, learner):
    data1 = [truth(sentence, True)]
    data2 = [truth(sentence, True), truth(sentence, True)]
    
    prob1 = learner.prob(data1)
    prob2 = learner.prob(data2)

    print data1

    assert prob1 >= 0.0
    assert prob1 <= 1.0
    assert prob1 == prob2

def test_bounds(sentence):
    data = [truth(sentence, True)]
    learner = Learner()
    learner.hidden_dims = 2
    learner.dims['noun'] = 3
    learner.dims['det'] = 4
    learner.dims['np'] = 5
    learner.initialise([data])
    prob = learner.prob(data)


def test_contradiction(sentence, learner):
    data = [truth(sentence, True), truth(sentence, False)]
    print data
    prob = learner.prob(data)
    assert prob == 0.0

def test_substitute_values(sentence, learner):
    data = (truth(sentence, True),)

    results = list(learner.substitute_values(data, {}))
    print "First ten expressions:"
    for x, subs in results[:10]:
        print x, subs
        assert len(subs) > 0
    
def test_substitute_values_with_duplicates(sentence, learner):
    data1 = (truth(sentence, True),)
    data2 = (truth(sentence, True), truth(sentence, True))
    
    results1 = list(learner.substitute_values(data1, {}))
    results2 = list(learner.substitute_values(data2, {}))
    assert len(results1) == len(results2)

def test_substitute_values_contradiction(sentence, learner):
    data = (truth(sentence, True), truth(sentence, False))
    
    results = list(learner.substitute_values(data, {}))
    assert len(results) == 0
    
def test_substitute_expression_word(learner):
    expression = ('w', 'det', 'some')
    values = list(learner.substitute_expression_values(expression, {}))

    assert len(values) == 2
    for e, subs in values:
        print e, subs
        assert len(subs) == 1
        assert e[:-1] == expression
        assert e[-1] in [0,1]
        
def test_substitute_expression_recursion(learner):
    expression = ('f', 'np', ('w', 'det', 'some'), ('w', 'noun', 'cats'))
    values = list(learner.substitute_expression_values(expression, {}))
    assert len(values) == 8
    for e, subs in values:
        print e, subs
        assert len(subs) == 3
        assert e[:2] == ('f', 'np')
        assert e[-1] in [0,1]
        assert e[2][-1] in [0,1]
        assert e[3][-1] in [0,1]

def test_substitute_expression_repeated_values(learner):
    expression = ('f', 'np', ('w', 'det', 'some'), ('w', 'det', 'some'))
    values = list(learner.substitute_expression_values(expression, {}))
    assert len(values) == 4

def test_gradient(learner, sentence):
    data = (truth(sentence, True),)

    prob_before = learner.prob(data)
    gradient = learner.ascend(data)
    prob_after = learner.prob(data)
    
    assert prob_after > prob_before
