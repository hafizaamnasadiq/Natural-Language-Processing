import pandas as pd
from evalMet import ConfusionMetrics

class Perception:

    def __init__(self, dataset):
        self.dataset = dataset
        self.vocab = self.build_vocab()
        self.w = [0] * len(self.vocab)
        self.b = 0

    def tokenize(self, text):
        # text -> list of tokens
        return text.lower().split()



    def build_vocab(self):
        vocab = {}

        for text, label in self.dataset:
            for word in self.tokenize(text):

                if word not in vocab:
                    vocab[word] = len(vocab)

        return vocab

    def vectorize(self, text):
        vector = [0] * len(self.vocab)

        for word in self.tokenize(text):
            if word in self.vocab:
                index = self.vocab[word]
                vector[index] = 1

        return vector

    def predict(self,x):
        labelScore = 0
        labelScore = sum(w*x for w, x in zip(self.w, x))
        # for w, xi in zip(self.w, x):
        #     labelScore += w * xi

        labelScore += self.b

        # check label if -1 & 0 -> normal ,, 1->spam
        if labelScore > 0:
            return 1
        else:
            return -1

    def update(self,x , lab):
        p = self.predict(x)

        if(p != lab):
            for i in range(len(self.w)):
                self.w[i] += lab * x[i]
            self.b +=lab
    
    def trainModel(self, epochs=10):
        for _ in range(epochs):
            for text, label in self.dataset:
                x = self.vectorize(text)
                self.update(x, label)

    def test(self, text):
        x = self.vectorize(text)
        prediction = self.predict(x)

        if prediction == 1:
            return "spam"
        else:
            return "normal"
        
    def test_dataset(self, test_dataset):
        results = []

        for text, label in test_dataset:
            x = self.vectorize(text)
            prediction = self.predict(x)

            if prediction > 0:
                pred_label = 1
            else:
                pred_label = -1
            results.append((label, pred_label))

        return results



# TEST WITH EXAMPLE SAMPLE
print("TEST WITH EXAMPLE SAMPLE")
dataset = [("free money now", 1),
          ("win money now", 1),        
          ("call me now", -1),         
          ("let's meet now", -1)]
perception = Perception(dataset)
perception.trainModel()
print(perception.test("free money"))
print(perception.test("call me"))
print(perception.test("meet now"))
print(perception.vocab)
print(perception.w)

# LOAD SENTIMENT DATASET, TEST TRAIN SEPARATE
print("TEST WITH SENTIMENTAL DATA SAMPLE")

df = pd.read_csv("IMDB Dataset.csv")
train_df = df.head(3000)
test_df = df.tail(500)
dataset = []
for _, row in train_df.iterrows():
    text = row["review"]
    if row["sentiment"] == "positive":
        label = 1
    else:
        label = -1
    dataset.append((text, label))
    
# Test data
test_data = []
for _, row in test_df.iterrows():
    text = row["review"]
    if row["sentiment"] == "positive":
        label = 1
    else:
        label = -1
    test_data.append((text, label))

# TRAINING 
perceptionModel = Perception(dataset)
print("Training...")
perceptionModel.trainModel()
print("Training Done!")

# TEST
print("Testing Start...")
test_res = perceptionModel.test_dataset(test_data)
print("Testing Done!")
print (test_res)
y_true = []
y_pred = []
for true, pred in test_res:
    y_true.append(true)
    y_pred.append(pred)
# pred_labels = [self.test(text) for text, _ in test_data]
# Evaluation
# Initialize the reused class

metrics = ConfusionMetrics(y_true, y_pred)

print(f"Accuracy: {metrics.accuracy()}")
print(f"Recall: {metrics.recall()}")
print(f"Precision: {metrics.precision()}")
print(f"F1 Score: {metrics.f1_score()}")





         


