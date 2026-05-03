from evalMet import ConfusionMetrics
import math

dataset = [
    ("free money now", "spam"),
    ("win money now", "spam"),
    ("call me now", "normal"),
    ("let's meet now", "normal")
]

def tokenize(text):
    # text -> list of tokens
    return text.lower().split()

def build_vocab(dataset):
    # all text > set of all the words we have
    vocab = set()
    for text, label in dataset:
        vocab.update(tokenize(text))
    return vocab

vocab = build_vocab(dataset)

# Initialize dictionaries to store word counts and class totals[cite: 1]
total_words = {"spam": 0, "normal": 0}
class_counts = {"spam": 0, "normal": 0}
word_counts = {"spam": {word: 0 for word in vocab}, 
               "normal": {word: 0 for word in vocab}}

# Fill dictionaries in a loop[cite: 1]
for text, label in dataset:
    class_counts[label] += 1
    tokens = tokenize(text)
    for word in tokens:
        word_counts[label][word] += 1
        total_words[label] += 1

# Compute priors: P(y)[cite: 1]
total_docs = len(dataset)
priors = {label: count / total_docs for label, count in class_counts.items()}

# Compute word likelihoods: P(word|label)[cite: 1]
likelihoods = {"spam": {}, "normal": {}}
vocab_size = len(vocab)

for label in ["spam", "normal"]:
    for word in vocab:
        # P(word|class) = (word_count + 1) / (total_words_in_class + vocab_size)[cite: 1]
        likelihoods[label][word] = (word_counts[label][word] + 1) / (total_words[label] + vocab_size)

def score(text, label):
    # Score calculation: P(c) * product of P(w|c)[cite: 1]
    tokens = tokenize(text)
    score_val = priors[label]
    for word in tokens:
        if word in likelihoods[label]:
            score_val *= likelihoods[label][word]
        else:
            # Handling words not in training vocab with smoothing[cite: 1]
            score_val *= 1 / (total_words[label] + vocab_size)
    return score_val

def predict(text):
    # Pick the highest score[cite: 1]
    spam_score = score(text, "spam")
    normal_score = score(text, "normal")
    
    return "spam" if spam_score > normal_score else "normal"

# Test the implementation[cite: 1]
print(predict("free money")) # Expected: spam[cite: 1]

# CONFUSION MATRIC
def confusion_metirc(test_data):
    
    y_true = [1 if label == "spam" else 0 for _, label in test_data]
    y_pred = [1 if predict(text) == "spam" else 0 for text, _ in test_data]

    # Initialize the reused class
    metrics = ConfusionMetrics(y_true, y_pred)

    print(f"Accuracy: {metrics.accuracy()}")
    print(f"Recall: {metrics.recall()}")
    print(f"Precision: {metrics.precision()}")
    print(f"F1 Score: {metrics.f1_score()}")

test_data= [("free money", "spam"), ("call me now", "normal")]
confusion_metirc(test_data)