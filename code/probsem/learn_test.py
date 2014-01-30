from probsem.learn import Learner, words, functions
import pytest
import logging
import sys

@pytest.fixture
def sentence():
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
    return f(s, f(np, w(det, 'some'), w(noun, 'cats')),
             f(vp, w(verb, 'like'), f(np, w(det, 'all'), w(noun, 'dogs'))))

@pytest.fixture
def learner():
    s = sentence()
    theory = [truth(s, True)]
    l = Learner()
    l.initialise(theory)
    return l

def truth(l, v):
    return ['truth', l, v]

def test_initialise(sentence):
    data = [truth(sentence, True)]
    print data

    learner = Learner()
    learner.initialise(data)
    print "Theta: ", learner.theta

    assert len(learner.theta[('w', 'det', 'some')]) > 0
    assert len(learner.theta[('f', 's', 'np', 'vp')]) > 0


def test_consistency(sentence, learner):
    data1 = [truth(sentence, True)]
    data2 = [truth(sentence, True), truth(sentence, True)]
    
    prob1 = learner.prob(data1)
    prob2 = learner.prob(data2)

    print data1

    assert prob1 >= 0.0
    assert prob1 <= 1.0
    assert prob1 == prob2

def test_contradiction(sentence, learner):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.debug("Starting test")
    data = [truth(sentence, True), truth(sentence, False)]
    print data
    prob = learner.prob(data)
    assert prob == 0.0

def test_substitute_values(sentence, learner):
    data = (truth(sentence, True),)
    
    learner = Learner()
    results = list(learner.substitute_values(data, {}))
    print "First ten expressions:"
    for x, subs in results[:10]:
        print x, subs
        assert len(subs) > 0
    
def test_substitute_values_with_duplicates(sentence, learner):
    data1 = (truth(sentence, True),)
    data2 = (truth(sentence, True), truth(sentence, True))
    
    learner = Learner()
    results1 = list(learner.substitute_values(data1, {}))
    results2 = list(learner.substitute_values(data2, {}))
    assert len(results1) == len(results2)

def test_substitute_values_contradiction(sentence, learner):
    data = (truth(sentence, True), truth(sentence, False))
    
    learner = Learner()
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

    
    
