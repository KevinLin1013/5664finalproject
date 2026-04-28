import pandas as pd


df = pd.read_csv("dataset/Truth_Seeker_Model_Dataset.csv")


df = df[['tweet', 'BinaryNumTarget']]
df = df.dropna()


df = df.sample(n=10000, random_state=42)


print(df.head())
print(df.shape)
print(df['BinaryNumTarget'].value_counts())

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["tweet"].tolist(),
    df["BinaryNumTarget"].astype(int).tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["BinaryNumTarget"]
)

print("Train size:", len(train_texts))
print("Test size:", len(test_texts))
print("First train text:", train_texts[0])
print("First label:", train_labels[0])

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128
)

print("Tokenization done")
print(train_encodings.keys())


import torch

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, train_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

print("Dataset ready")
print(len(train_dataset), len(test_dataset))

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_dir="./logs"
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import numpy as np

# Get predictions on test set
predictions = trainer.predict(test_dataset)

logits = predictions.predictions
true_labels = predictions.label_ids

# Predicted class
pred_labels = np.argmax(logits, axis=1)

# Probability for positive class
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
positive_probs = probs[:, 1]

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# ROC-AUC
roc_auc = roc_auc_score(true_labels, positive_probs)

print("\nConfusion Matrix:")
print(cm)

print("\nROC-AUC:")
print(roc_auc)

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, digits=4))
