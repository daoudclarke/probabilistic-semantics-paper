# Bismillahi-r-Rahmani-r-Rahim
#
# Analyse output of Montague semantics learning



def analyse(f):
    correct = 0
    incorrect = 0
    for line in f:
        text, hypothesis, entailment, t_prob, th_prob = line.strip().split('\t')
        entailment = (entailment == 'True')
        t_prob, th_prob = float(t_prob), float(th_prob)
        judgment = th_prob/t_prob > 0.75
        if judgment == entailment:
            correct += 1
        else:
            incorrect += 1
        print entailment, judgment, t_prob, th_prob

    print "Correct: ", correct
    print "Incorrect: ", incorrect
    print "Proportion correct: ", correct/float(correct + incorrect)


if __name__ == "__main__":
    f = open('output.csv')
    analyse(f)
