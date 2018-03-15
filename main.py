import pandas as pd

from MultimodalNaiveBayes import MultimodalNaiveBayes

train_data = pd.read_csv('BlogSentences.txt', sep="	")
train_data.columns = ["Target", "Text"]

model = MultimodalNaiveBayes()

model.train(train_data)
error_rate = model.test(train_data)
print(error_rate)
