# Bismillahi-r-Rahmani-r-Rahim
#
# Create datasets to test out probabilistic Montague semantics

from random import Random

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

WOMEN = ['housewives', 'ladies', 'maidens', 'mistresses', 'whores', 'widows', 'wives']
EMPLOYEES = ['busboys', 'cowboys', 'laborers', 'movers', 'punchers', 'salesmen', 'waiters', 'workmen']
QUANTIFIERS = [('few', 'some'),
               ('all', 'most'),
               ('most', 'some'),
               ('all', 'some'),
               ('ten', 'some'),
               ('some', 'some'),
               ('all', 'all'),
               ('no', 'no')]
INCREASING = ['some', 'ten']
DECREASING = ['all', 'no']
NONINCREASING = ['few', 'most']

def make_small_dataset():
    pairs = (zip(WOMEN, ['women']*len(WOMEN)) +
             zip(EMPLOYEES, ['employees']*len(EMPLOYEES)))
    pattern = "like %s %s the sun"
    random = Random(1)
    dataset = []
    for text_noun, hypothesis_noun in pairs:
        alternative_noun = 'employees' if hypothesis_noun == 'women' else 'women'
        for text_quantifier, hypothesis_quantifier in QUANTIFIERS:
            if text_quantifier in INCREASING or hypothesis_quantifier in INCREASING:
                if random.choice([True,False]):
                    dataset.append( (pattern % (text_quantifier, text_noun),
                                     pattern % (hypothesis_quantifier, hypothesis_noun),
                                     True) )
                else:
                    if text_quantifier not in DECREASING and hypothesis_quantifier not in DECREASING:
                        dataset.append( (pattern % (text_quantifier, hypothesis_noun),
                                         pattern % (hypothesis_quantifier, text_noun),
                                         False) )
                    else:
                        dataset.append( (pattern % (text_quantifier, text_noun),
                                         pattern % (hypothesis_quantifier, alternative_noun),
                                         False) )
            if text_quantifier in DECREASING or hypothesis_quantifier in DECREASING:
                if random.choice([True,False]):
                    dataset.append( (pattern % (text_quantifier, hypothesis_noun),
                                     pattern % (hypothesis_quantifier, text_noun),
                                     True) )
                else:
                    if text_quantifier not in INCREASING and hypothesis_quantifier not in INCREASING:
                        dataset.append( (pattern % (text_quantifier, text_noun),
                                         pattern % (hypothesis_quantifier, hypothesis_noun),
                                         False) )
                    else:
                        dataset.append( (pattern % (text_quantifier, alternative_noun),
                                         pattern % (hypothesis_quantifier, text_noun),
                                         False) )
            if text_quantifier in NONINCREASING:
                assert text_quantifier != hypothesis_quantifier
                if random.choice([True,False]):
                    dataset.append( (pattern % (text_quantifier, text_noun),
                                     pattern % (hypothesis_quantifier, text_noun),
                                     True) )                    
                else:
                    dataset.append( (pattern % (text_quantifier, hypothesis_noun),
                                     pattern % (text_quantifier, text_noun),
                                     False) )                    
    return dataset


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
    tests = []
    for text, hypothesis, entails in dataset:
        text_sentence = make_sentence(text)
        hypothesis_sentence = make_sentence(hypothesis)
        tests.append( (text_sentence, hypothesis_sentence,
                       "'%s\t%s\t%s'" % (text, hypothesis, str(entails))) )
    return ":- test([%s])." % ', '.join('[%s, %s, %s]' % x for x in tests)

if __name__ == "__main__":
    print ":- prism([learn_mode=ml, default_sw_d=0.0, epsilon=1.0e-10, restart=1], montague)."
    #print make_data(QUANTIFIER)
    #print make_test(QUANTIFIER_TEST)
    #print make_data(LEXICAL)
    #print make_test(LEXICAL_TEST)

    dataset = make_small_dataset()
    random = Random(2)
    random.shuffle(dataset)

    train_length = int(len(dataset)*0.66)
    train = dataset[:train_length]
    test = dataset[train_length:]

    print make_data(train)
    print make_test(test)
    
