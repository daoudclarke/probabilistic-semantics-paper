# Bismillahi-r-Rahmani-r-Rahim
#
# Create datasets to test out probabilistic Montague semantics

SENTENCE = "f(s, f(vp, w(%s, verb), f(np, w(%s, det), w(%s, noun))), f(np, w(%s, det), w(%s, noun)))"

POSITIVE = ["count(theory([truth(%s, t1), truth(%s, t1)]), 1)",
            "count(theory([truth(%s, t0), truth(%s, t1)]), 1)",
            "count(theory([truth(%s, t0), truth(%s, t0)]), 1)"]

NEGATIVE = ["count(theory([truth(%s, t1), truth(%s, t0)]), 1)"]

# LEXICAL = [('chased the cat the dog', 'chased the animal the dog', True)]

# LEXICAL_TEST = [('chased the cat the dog', 'chased the animal the dog')]

LEXICAL = [('chased the cat the dog', 'chased the animal the dog', True),
           ('likes the cat the dog', 'likes the animal the dog', True)]

LEXICAL_TEST = [('likes the cat the dog', 'likes the animal the dog'),
                ('chased the animal the dog', 'chased the cat the dog'),
                ('loves the cat the dog', 'loves the animal the dog'),
                ('loves the animal the dog', 'loves the cat the dog')]

QUANTIFIER = [('like some cats all dogs','like some animals all dogs', True),
              ('like no animals all dogs','like no cats all dogs', True),
              ('like some men all dogs','like some people all dogs', True)]


QUANTIFIER_TEST = [('like no people all dogs','like no men all dogs'),
                   ('like no men all dogs','like no people all dogs'),
                   ('like most people all dogs','like most men all dogs')]

# QUANTIFIER_TEST = [('love no people all apples','love no men all apples'),
#                    ('love no men all apples','love no people all apples'),
#                    ('love most people all apples','love most men all apples')]


def make_sentence(sentence):
    return SENTENCE % tuple(sentence.split())

def make_data(dataset):
    result = []
    for text, hypothesis, entails in dataset:
        text_sentence = make_sentence(text)
        hypothesis_sentence = make_sentence(hypothesis)
        sentences = POSITIVE if entails else NEGATIVE
        for s in sentences:
            result.append(s % (text_sentence, hypothesis_sentence))
    return ":- learn([%s])." % ', '.join(result)

def make_test(dataset):
    result = ''
    for text, hypothesis in dataset:
        text_sentence = make_sentence(text)
        hypothesis_sentence = make_sentence(hypothesis)
        result += ":- writeln('%s').\n" % (text + ' -> ' + hypothesis)
        result += ':- prob(theory([truth(%s, t1)]), X), writeln(X).\n' % text_sentence
        result += ':- prob(theory([truth(%s, t1), truth(%s, t1)]), X), writeln(X).\n' % (text_sentence, hypothesis_sentence)
    return result

if __name__ == "__main__":
    print ":- prism([learn_mode=ml, default_sw_d=0.0, epsilon=1.0e-10, restart=10], montague)."
    #print make_data(QUANTIFIER)
    #print make_test(QUANTIFIER_TEST)
    print make_data(LEXICAL)
    print make_test(LEXICAL_TEST)
