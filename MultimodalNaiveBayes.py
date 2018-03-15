import operator
import re
import string


class MultimodalNaiveBayes:
    def __init__(self):
        self.model = {}
        self.targets = []
        self.num_terms = 0
        self.target_terms = {}

    def get_terms(self, text):
        clean_text = re.sub('[' + string.punctuation + ']', '', text)
        clean_text = clean_text.lower()
        terms = clean_text.split()
        return terms

    def train(self, train_data):
        self.model = {}
        self.targets = list(set(train_data['Target']))
        i = 0
        for index, row in train_data.iterrows():
            terms = self.get_terms(row['Text'])
            for term in terms:
                i += 1
                self.add_term_to_model(term, row['Target'])

        self.num_terms = len(self.model.keys())
        for target in self.targets:
            self.target_terms[target] = sum([self.get_occurrences(term, target) for term in self.model.keys()])

    def add_term_to_model(self, term, target):
        if term not in self.model:
            self.model[term] = {}
        if target not in self.model[term]:
            self.model[term][target] = 0
        self.model[term][target] += 1

    def predict(self, text):
        terms = self.get_terms(text)
        result = {}
        for target in self.targets:
            target_result = 1
            for term in terms:
                target_result *= self.get_probability(target, term)
            result[target] = target_result
        return result

    def get_occurrences(self, term, target):
        if term not in self.model:
            return 0
        if target not in self.model[term]:
            return 0
        return self.model[term][target]

    def get_probability(self, target, term):
        return (self.get_occurrences(term, target) + 1)/(self.target_terms[target] + self.num_terms)

    def test(self, test_data):
        errors = 0
        total = 0
        for index, row in test_data.iterrows():
            prediction = self.predict(row['Text'])
            pred_target = max(prediction.items(), key=operator.itemgetter(1))[0]
            if pred_target != row['Target']:
                errors += 1
            total += 1
        return errors/total


