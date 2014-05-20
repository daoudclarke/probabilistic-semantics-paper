# Bismillahi-r-Rahmani-r-Rahim

# Evaluate phrase entailment using machine learning

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

import csv

class Experiment(object):
    def __init__(self, train_name, test_name, output_name):
        self.train_name = train_name
        self.test_name = test_name
        self.output_name = output_name

    def train(self):
        train = self.read(self.train_name)
        train_features = self.features(train)
        self.vectorizer = DictVectorizer(sparse=False)
        train_matrix = self.vectorizer.fit_transform(train_features)
        values = [x[2]=='True' for x in train]
        assert set(values) == set([True, False])
        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(train_matrix, values)

    def test(self):
        test = self.read(self.test_name)
        test_features = self.features(test)
        test_matrix = self.vectorizer.transform(test_features)
        values = self.classifier.predict(test_matrix)
        return [x + [y] for x,y in zip(test, values)]

    def read(self, file_name):
        with open(file_name) as data_file:
            reader = csv.reader(data_file, delimiter='\t')
            return [x[:3] for x in reader]


    def features(self, data):
        for row in data:
            text, hypothesis, entailment = row
            features = {'T_' + x:1. for x in text.split()}
            features.update({'H_' + x:1. for x in hypothesis.split()})
            yield features

    def output(self, results):
        print results
        with open(self.output_name, 'wb') as output_file:
            writer = csv.writer(output_file, delimiter='\t')
            for text, hypothesis, entailment, judgment in results:
                value = 1.0 if judgment else 0.0
                row = [text, hypothesis, entailment, 1.0, value]
                writer.writerow(row)
            
    def run(self):
        self.train()
        results = self.test()
        self.output(results)

if __name__ == "__main__":
    experiment = Experiment('train.csv', 'test.csv', 'test_ml.csv')
    experiment.run()
