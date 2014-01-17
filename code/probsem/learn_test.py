

from probsem.learn import Learner, words, functions

def test_learn():
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
    def truth(l, v):
        return ['truth', l, v]
    data = [truth(f(s, f(np, w(det, 'some'), w(noun, 'cats')), f(vp, w(verb, 'like'), f(np, w(det, 'all'), w(noun, 'dogs')))), True)]
    print data

    learner = Learner()
    learner.initialise(data)
    print learner.theta
    assert len(learner.theta[('det', 'some')]) > 0



if __name__ == "__main__":
    test_learn()
