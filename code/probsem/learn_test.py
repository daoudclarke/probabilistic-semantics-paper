from probsem.learn import Learner, words, functions
from pytest import fixture

@fixture
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

def truth(l, v):
    return ['truth', l, v]

def test_learn(sentence):
    data = [truth(sentence, True)]
    print data

    learner = Learner()
    learner.initialise(data)
    print learner.theta

    assert len(learner.theta[('det', 'some')]) > 0


def test_consistency(sentence):
    data1 = [truth(sentence, True)]
    data2 = [truth(sentence, True), truth(sentence, True)]
    
    learner = Learner()
    learner.initialise(data1)

    prob1 = learner.prob(data1)
    prob2 = learner.prob(data2)

    assert prob1 >= 0.0
    assert prob1 <= 1.0
    assert prob1 == prob2
