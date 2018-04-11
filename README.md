# Multimodal Naive Bayes

This project contains an implementation of the Multimodal Naive Bayes algorithm (**MultimodalNaiveBayes.py**), used to classification problems for text datasets. The project also contains a python script (**main.py**) which is an example of the use of the algorithm and four datasets to test it (**data.zip**).

## MultimodalNaiveBayes.py

This section contains the descriptions of every method contained in the *MultimodalNaiveBayes* class.

### train

```python
def train(self, train_data):
    # Init self.targets
    self.targets = list(set(train_data['Target']))

    # Init self.model
    self.model = {}
    i = 0
    for index, row in train_data.iterrows():
        terms = self.get_terms(row['Text'])
        for term in terms:
            i += 1
            self.add_term_to_model(term, row['Target'])

    # Init self.num_terms
    self.num_terms = len(self.model.keys())

    # Init self.target_terms
    for target in self.targets:
        self.target_terms[target] = sum([self.get_occurrences(term, target) for term in self.model.keys()])
```

The *train* method gets a parameter *train_data* that contains a dataset with the data and train the model. This dataset has to contain two columns:
* **Target:** Represents the prediction variable for each row.
* **Text:** Contains the text used to predict the *Target* value.

Given this, the method uses the train data to initialize the four class variables:
* **targets:** It is an array that contains all the different target values of the train data.
* **model:** The model variable is a dictionary that contains, for each term and for each target, the number of occurrences (if any) of the term in a text related with the target value.
* **num_terms:** The number of different terms contained in the train data.
* **target_terms:** It is a dictionary that contains, for each target value, the number of the terms that contain the texts related to the target value.

#### *Example:*
* ##### *Train data:*
| Text                             | Target   |
|----------------------------------|----------|
| The mic is great.                | Positive |
| Works great!.                    | Positive |
| What a waste of money and time!. | Negative |
| Mic Doesn't work.                | Negative |

* ##### *Initial class variables values:*
```python
targets = ['Positive', 'Negative']
model = {
  'the': {
    'Positive': 1
  },
  'mic': {
    'Positive': 1,
    'Negative': 1
  },
  ...
}
num_terms = 14
target_terms = {
  'Positive': 6,
  'Negative': 10
}
```

### get_terms

```python
def get_terms(self, text):
    clean_text = re.sub('[' + string.punctuation + ']', '', text)
    clean_text = clean_text.lower()
    terms = clean_text.split()
    return terms
```

The *get_terms* method gets a parameter *text* that contains a string that represents the text of a row. The method removes the punctuation characters, transform the text to lowercase and return an array with all the terms of the text.

### add_term_to_model

```python
def add_term_to_model(self, term, target):
    if term not in self.model:
        self.model[term] = {}
    if target not in self.model[term]:
        self.model[term][target] = 0
    self.model[term][target] += 1
```

The *add_term_to_model* method gets a parameter *term* and a parameter *target* and adds to the *model* variable an occurrence of the term in a text related to the target.

### get_occurrences

```python
def get_occurrences(self, term, target):
    if term not in self.model:
        return 0
    if target not in self.model[term]:
        return 0
    return self.model[term][target]
```

The *get_occurrences* method gets a parameter *term* and a parameter *target* and return the number of occurrences of the term in texts related to the target.

### test

```python
def test(self, test_data):
    errors = 0
    total = 0
    for index, row in test_data.iterrows():
        prediction = self.predict(row['Text'])
        pred_target = max(prediction.items(), key=operator.itemgetter(1))[0]
        if pred_target != row['Target']:
            errors += 1
        total += 1
    return (errors / total) * 100
```

The *test* method gets a parameter *test_data*, in the same format than the parameter *train_data* in the *test* method, and returns the error rate of the predictions for these new instances using the previous train.

### predict

```python
def predict(self, text):
    terms = self.get_terms(text)
    result = {}
    for target in self.targets:
        target_result = 1
        for term in terms:
            target_result *= self.get_probability(target, term)
        result[target] = target_result
    return result
```

The *predict* method gets a parameter *text* and returns the predicted target to this text. This prediction is based on the target with the highest probability. This target probability is based on the probability of every term in the text for the target. Concretely, the target probability is the product of the all term probabilities for the target.

### get_probability

```python
def get_probability(self, target, term):
    return (self.get_occurrences(term, target) + 1) / (self.target_terms[target] + self.num_terms)
```

The *get_probability* method gets a parameter *term* and a parameter *target* and computes the probability of this target given this term. In order to avoid problems when the text to predict contains some term that, given the target, any related train text doesn't contain, the method uses Laplace smoothing. The basic probability should be the number of occurrences of the term for the target divided by the number of occurrences of all the terms for the target. In order to apply the Laplace smoothing, a fictitious occurrence is added to the nominator and the total number of occurrences of all terms is added to the denominator. This modifies the probabilities but keeps the relation between the different probabilities. So, it is equivalent if we are basing our prediction on the highest probability.

### show

```python
def show(self):
    print("Term probs:")
    terms_occ = {}
    for term in self.model.keys():
        term_occurrences = sum([self.get_occurrences(term, target) for target in self.targets])
        terms_occ[term] = term_occurrences
    for term, term_occurrences in sorted(terms_occ.items(), key=lambda x: x[1], reverse=True):
        print("Term", "\"" + term + "\"", "appears", term_occurrences, "times", ":")
        for target in self.targets:
            print(target, ":", round(self.get_occurrences(term, target) / term_occurrences, 3))
```

The *show* method prints the number of occurrences of each term in the training dataset and how they are distributed among the targets.

#### *Example:*

Assuming that the targets are 0 and 1:

```
Term probs:
Term "the" appears 432 times :
0 : 0.454
1 : 0.546
Term "and" appears 291 times :
0 : 0.433
1 : 0.567
...
```

## main.py

The *main.py* file contains a python script that shows how the MultimodalNaiveBayes works. The comments explain how it works. Commenting and uncommenting the read dataset lines, we can select which dataset we want to use.

```python
import pandas as pd
from MultimodalNaiveBayes import MultimodalNaiveBayes

## Set a seed to allow the experiment replications
seed = 24011994

## Read the dataset
data = pd.read_csv('data/BlogSentences.txt', sep="	", names=["Target", "Text"])
#data = pd.read_csv('data/AmazonComments.txt', sep="	", names=["Text", "Target"])
#data = pd.read_csv('data/IMDBComments.txt', sep="	", names=["Text", "Target"])
#data = pd.read_csv('data/YelpComments.txt', sep="	", names=["Text", "Target"])

## Split the dataset between train (75% of data) and test (25% of data)
train_data = data.sample(frac=0.75, random_state=seed)
test_data = data.drop(train_data.index)

## Instance and initialize the model using the train data
model = MultimodalNaiveBayes()
model.train(train_data)

## Compute the error rate based on the test data
error_rate = model.test(test_data)
print("Error rate", error_rate)

## Print the distribution of the term occurrences in the dataset
model.show()
```

## data.zip

This compressed file contains the four datasets that the *main.py* is prepared to read. The *main.py* file assumes that they are unzipped in a *data* folder. The datasets are not added to the repository in order to avoid saturation on the repository. The final project structure should be:

- MultimodaNaiveBayes
    - data
        - AmazonComments.txt
        - BlogSentences.txt
        - IMDBComments.txt
        - YelpComments.txt
    - data.zip
    - main.py
    - MultimodalNaiveBayes.py
    - README.md
        
