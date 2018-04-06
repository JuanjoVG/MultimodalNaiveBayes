import pandas as pd

from MultimodalNaiveBayes import MultimodalNaiveBayes
seed = 24011994

#data = pd.read_csv('BlogSentences.txt', sep="	")
#data.columns = ["Target", "Text"]

#data = pd.read_csv('data/AmazonComments.txt', sep="	")
#data = pd.read_csv('data/IMDBComments.txt', sep="	")
data = pd.read_csv('data/YelpComments.txt', sep="	")
data.columns = ["Text", "Target"]

print(data)
train_data = data.sample(frac=0.75, random_state=seed)
test_data = data.drop(train_data.index)

model = MultimodalNaiveBayes()

model.train(train_data)
error_rate = model.test(test_data)
print("Error rate", error_rate)
model.show()
