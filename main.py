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
