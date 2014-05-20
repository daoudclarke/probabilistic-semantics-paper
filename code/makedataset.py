# Bismillahi-r-Rahmani-r-Rahim
#
# Create datasets to test out probabilistic Montague semantics

from random import Random

SENTENCE = "f(s, f(np, w(%s, det), w(%s, noun)), f(vp, w(%s, verb), f(np, w(%s, det), w(%s, noun))))"

POSITIVE = ["count(theory([truth(%s, t1), truth(%s, t1)]), 1)",
            "count(theory([truth(%s, t0), truth(%s, t1)]), 1)",
            "count(theory([truth(%s, t0), truth(%s, t0)]), 1)"]

NEGATIVE = ["count(theory([truth(%s, t1), truth(%s, t0)]), 1)"]

# LEXICAL = [('chased the cat the dog', 'chased the animal the dog', True)]

# LEXICAL_TEST = [('chased the cat the dog', 'chased the animal the dog')]

LEXICAL = [('the cat chased the dog', 'the animal chased the dog', True),
           ('the cat likes the dog', 'the animal likes the dog', True)]

LEXICAL_TEST = [('the cat likes the dog', 'the animal likes the dog', True),
                ('the animal chased the dog', 'the cat chased the dog', False),
                ('the cat loves the dog', 'the animal loves the dog', True),
                ('the animal loves the dog', 'the cat loves the dog', False)]

QUANTIFIER = [('some cats like all dogs','some animals like all dogs', True),
              ('no animals like all dogs','no cats like all dogs', True),
              #('some dogs like all dogs', 'some animals like all dogs', True),
              #('no animals like all dogs', 'no dogs like all dogs', True),
              ('some men like all dogs','some people like all dogs', True)]

QUANTIFIER_TEST = [('no people like all dogs','no men like all dogs', True),
                   ('no men like all dogs','no people like all dogs', False),
                   ('most people like all dogs','most men like all dogs', False)]

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
    pattern = "%s %s like the sun"
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

def make_test(filename, dataset):
    tests = []
    for text, hypothesis, entails in dataset:
        text_sentence = make_sentence(text)
        hypothesis_sentence = make_sentence(hypothesis)
        tests.append( (text_sentence, hypothesis_sentence,
                       "'%s\t%s\t%s'" % (text, hypothesis, str(entails))) )
    return ":- test(%s, [%s])." % ("'%s'" % filename,
                                   ', '.join('[%s, %s, %s]' % x for x in tests))

if __name__ == "__main__":
    # print ":- set_prism_flag(daem,on)."
    # print ":- set_prism_flag(itemp_init,0.3)."
    # print ":- set_prism_flag(itemp_rate,1.2)."

    print ":- prism([epsilon=1.0e-5, restart=20], montague)."
    # print make_data(QUANTIFIER)
    # print make_test('train.csv', QUANTIFIER)
    # print make_test('test.csv', QUANTIFIER_TEST)
    #print make_data(LEXICAL)
    #print make_test(LEXICAL_TEST)

    dataset = make_small_dataset()
    random = Random(2)
    random.shuffle(dataset)

    train_length = int(len(dataset)*0.66)
    train = dataset[:train_length]
    test = dataset[train_length:]

    print make_data(train)
    print make_test('train.csv', train)
    print make_test('test.csv', test)
    print ":- halt."
