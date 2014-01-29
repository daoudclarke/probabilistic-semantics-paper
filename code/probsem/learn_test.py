from probsem.learn import Learner, words, functions
import pytest


@pytest.fixture
def sentence():
    s = 's'
    det = 'det'
    noun = 'noun'
    verb = 'verb'
    np = 'np'
    vp = 'vp'
    def f(p, l1, l2):
        return ['f', p, l1, l2]
    def w(w, p):
        return ['w', w, p]
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
    print "Words: ", learner.words
    print "Functions: ", learner.functions

    assert len(learner.words[('det', 'some')]) > 0
    assert len(learner.functions[('s', 'np', 'vp')]) > 0


@pytest.mark.xfail
def test_consistency(sentence, learner):
    data1 = [truth(sentence, True)]
    data2 = [truth(sentence, True), truth(sentence, True)]
    
    prob1 = learner.prob(data1)
    prob2 = learner.prob(data2)

    print data1

    assert prob1 >= 0.0
    assert prob1 <= 1.0
    assert prob1 == prob2

@pytest.mark.xfail
def test_substitute_values(sentence, learner):
    data = [truth(sentence, True), truth(sentence, True)]
    
    learner = Learner()
    for s in learner.substitute_values(data1):
        print s
    
def test_substitute_expression_word(learner):
    expression = ['w', 'det', 'some']
    values = list(learner.substitute_expression_values(expression, {}))

    assert len(values) == 2
    for e, subs in values:
        print e, subs
        assert len(subs) == 1
        assert e[:-1] == expression
        assert e[-1] in [0,1]
        
def test_substitute_expression_recursion(learner):
    expression = ['f', 'np', ['w', 'det', 'some'], ['w', 'noun', 'cats']]
    values = list(learner.substitute_expression_values(expression, {}))
    assert len(values) == 8
    for e, subs in values:
        print e, subs
        assert len(subs) == 3
        assert e[:2] == ['f', 'np']
        assert e[-1] in [0,1]
        assert e[2][-1] in [0,1]
        assert e[3][-1] in [0,1]

    
