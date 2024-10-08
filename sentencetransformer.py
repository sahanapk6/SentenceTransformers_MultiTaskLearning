# -*- coding: utf-8 -*-
"""SentenceTransformer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ogqmkHBIcC1CeeBzKW3pIZJ70hUVdk6Q
"""

import random
from sklearn.model_selection import train_test_split

def read_sentences_from_file(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        sentences = file.readlines()
    return sentences

file1 = "negative.txt"
file2 = "positive.txt"

sentences_from_file1 = read_sentences_from_file(file1)
sentences_from_file2 = read_sentences_from_file(file2)

labeled_list = []

# Randomly select sentences from both files and assign labels alternately
while sentences_from_file1 or sentences_from_file2:
    if sentences_from_file1:
        sentence = random.choice(sentences_from_file1)
        labeled_list.append((sentence.strip(), 1))  # Label for file 1
        sentences_from_file1.remove(sentence)

    if sentences_from_file2:
        sentence = random.choice(sentences_from_file2)
        labeled_list.append((sentence.strip(), 0))  # Label for file 2
        sentences_from_file2.remove(sentence)

random.shuffle(labeled_list)

# Extract labels and sentences
labels = [label for _, label in labeled_list]
sentences = [sentence for sentence, _ in labeled_list]

# Split the data into training and testing sets
sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# print("List of labels:")
# print(labels)
# print("\nList of corresponding sentences:")
# print(sentences)

from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.shared_linear = nn.Linear(self.bert.config.hidden_size, 256)

        # Task A: Sentence Classification
        self.classification_head = nn.Linear(256, 2)  # Assuming 2 classes: Positive, Negative

        # Task B: Sentiment Analysis
        self.sentiment_head = nn.Linear(256, 1)  # Regression head for sentiment score

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]  # CLS token output
        shared_embedding = self.shared_linear(cls_output)
        shared_embedding = F.normalize(shared_embedding, p=2, dim=1)

        # Task A: Classification
        logits = self.classification_head(shared_embedding)

        # Task B: Sentiment Analysis
        sentiment_score = self.sentiment_head(shared_embedding)

        return logits, sentiment_score

    def encode(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            shared_embedding = self.forward(input_ids, attention_mask)[0]  # Only take shared_embedding
        return shared_embedding

    def predict(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        logits, sentiment_scores = self.forward(input_ids, attention_mask)

        # For classification, apply softmax to get probabilities and then argmax for the predicted label
        probs = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probs, dim=1)

        return predicted_labels, sentiment_scores.squeeze()

model = MultiTaskSentenceTransformer()

from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
import torch.optim as optim

# Tokenize and create DataLoader
inputs = model.tokenizer(sentences_train, padding=True, truncation=True, return_tensors='pt')
train_dataset = TensorDataset(
    inputs['input_ids'], inputs['attention_mask'],
    torch.tensor(labels_train, dtype=torch.long)
)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Define loss functions
classification_loss_fn = nn.CrossEntropyLoss()
sentiment_loss_fn = nn.MSELoss()

# Training loop
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels_classification = batch
        optimizer.zero_grad()

        logits, sentiment_scores = model(input_ids, attention_mask)

        classification_loss = classification_loss_fn(logits, labels_classification)
        sentiment_loss = sentiment_loss_fn(sentiment_scores.squeeze(), labels_classification.float())

        loss = classification_loss + sentiment_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Sample sentences for testing
# test_sentences = [
#     "I love this product, it's amazing!",
#     "This is the worst experience I've ever had.",
#     "the worst film of the year . ",
#     "morton is , as usual , brilliant ."
# ]
# test_labels = [0, 1, 1, 0]

# Predict labels and sentiment scores
model.eval()
with torch.no_grad():
    predicted_labels, sentiment_scores = model.predict(sentences_test)

correct_predictions = (predicted_labels == torch.tensor(labels_test)).sum().item()
accuracy = correct_predictions / len(labels_test)

# Print the outputs
label_names = ["Positive", "Negative"]
print("Predicted Labels and Sentiment Scores:")
for i, sentence in enumerate(sentences_test):
    print(f"Sentence: {sentence}")
    print(f"Predicted Label: {label_names[predicted_labels[i].item()]}")
    print(f"Sentiment Score: {sentiment_scores[i].item()}")
    print()

print(f"Classification Accuracy: {accuracy * 100:.2f}%")

Transfer Learning

# Freeze the transformer backbone and Unfreeze the shared linear layer and task-specific heads
model = MultiTaskSentenceTransformer()

for param in model.bert.parameters():
    param.requires_grad = False
for param in model.shared_linear.parameters():
    param.requires_grad = True
for param in model.classification_head.parameters():
    param.requires_grad = True
for param in model.sentiment_head.parameters():
    param.requires_grad = True

# Define optimizer only for the unfrozen parameters
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)

# Define loss functions
classification_loss_fn = nn.CrossEntropyLoss()
sentiment_loss_fn = nn.MSELoss()

# Training loop
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels_classification = batch
        optimizer.zero_grad()

        logits, sentiment_scores = model(input_ids, attention_mask)

        classification_loss = classification_loss_fn(logits, labels_classification)
        sentiment_loss = sentiment_loss_fn(sentiment_scores.squeeze(), labels_classification.float())

        loss = classification_loss + sentiment_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Testing and evaluation remain the same
model.eval()
with torch.no_grad():
    predicted_labels, sentiment_scores = model.predict(sentences_test)

correct_predictions = (predicted_labels == torch.tensor(labels_test)).sum().item()
accuracy = correct_predictions / len(labels_test)

# Print the outputs
label_names = ["Positive", "Negative"]

print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# Freeze the transformer backbone,Freeze the classification head (Task A) Unfreeze the shared linear layer and sentiment head (Task B)
model = MultiTaskSentenceTransformer()

for param in model.bert.parameters():
    param.requires_grad = False
for param in model.shared_linear.parameters():
    param.requires_grad = True
for param in model.sentiment_head.parameters():
    param.requires_grad = True
for param in model.classification_head.parameters():
    param.requires_grad = False

# Define optimizer only for the unfrozen parameters
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5
)

# Define loss functions
classification_loss_fn = nn.CrossEntropyLoss()
sentiment_loss_fn = nn.MSELoss()

# Training loop
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels_classification = batch
        optimizer.zero_grad()

        logits, sentiment_scores = model(input_ids, attention_mask)

        classification_loss = classification_loss_fn(logits, labels_classification)
        sentiment_loss = sentiment_loss_fn(sentiment_scores.squeeze(), labels_classification.float())

        loss = classification_loss + sentiment_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Predict labels and sentiment scores
model.eval()
with torch.no_grad():
    predicted_labels, sentiment_scores = model.predict(sentences_test)

correct_predictions = (predicted_labels == torch.tensor(labels_test)).sum().item()
accuracy = correct_predictions / len(labels_test)

print(f"Classification Accuracy: {accuracy * 100:.2f}%")

"""Layer-wise Learning Rate Implementation"""

# Modify the optimizer to have different learning rates for different layers
optimizer = optim.AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},  # Lower learning rate for BERT layers
    {'params': model.shared_linear.parameters(), 'lr': 5e-4},  # Higher learning rate for the shared linear layer
    {'params': model.classification_head.parameters(), 'lr': 5e-4},  # Higher learning rate for classification head
    {'params': model.sentiment_head.parameters(), 'lr': 5e-4}  # Higher learning rate for sentiment head
])

# Training loop with layer-wise learning rates
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels_classification = batch
        optimizer.zero_grad()

        logits, sentiment_scores = model(input_ids, attention_mask)

        classification_loss = classification_loss_fn(logits, labels_classification)
        sentiment_loss = sentiment_loss_fn(sentiment_scores.squeeze(), labels_classification.float())

        loss = classification_loss + sentiment_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

model.eval()
with torch.no_grad():
    predicted_labels, sentiment_scores = model.predict(sentences_test)

correct_predictions = (predicted_labels == torch.tensor(labels_test)).sum().item()
accuracy = correct_predictions / len(labels_test)

print(f"Classification Accuracy: {accuracy * 100:.2f}%")