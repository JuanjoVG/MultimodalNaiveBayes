import pandas as pd

data = pd.read_csv('BlogSentences.txt', sep="	", header=None)
data.columns = ["Target", "Sentence"]

## Clean sentences
for sentence in data.Sentence:
    print(sentence.split())
