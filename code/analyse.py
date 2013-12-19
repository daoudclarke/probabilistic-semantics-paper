# Bismillahi-r-Rahmani-r-Rahim
#
# Analyse output of Montague semantics learning





def data(f):
    for line in f:
        text, hypothesis, entailment, t_prob, th_prob = line.strip().split('\t')
        entailment = (entailment == 'True')
        t_prob, th_prob = float(t_prob), float(th_prob)
        if t_prob == 0.0:
            degree = 0.5
        else:
            degree = th_prob/t_prob
        yield entailment, degree

def get_threshold(data):
    best_threshold = 0.0
    best_accuracy = 0.0
    for threshold in [0.01*x for x in range(100)]:
        accuracy = analyse(data, threshold)
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy
    print "Best threshold", best_threshold
    print "Best accuracy", best_accuracy
    return best_threshold

def analyse(data, threshold):
    correct = 0
    incorrect = 0
    for entailment, degree in data:
        judgment = (degree > threshold)
        if judgment == entailment:
            correct += 1
        else:
            incorrect += 1
        #print entailment, judgment, degree

    return correct/float(correct + incorrect)


if __name__ == "__main__":
    with open('train.csv') as train_file:
        train = list(data(train_file))
    threshold = get_threshold(train)
    with open('test.csv') as test_file:
        test = list(data(test_file))
    result = analyse(test, threshold)
    print "Accuracy: ", result
