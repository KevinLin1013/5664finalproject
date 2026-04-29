import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from gensim.downloader import load


df = pd.read_csv("dataset/Truth_Seeker_Model_Dataset.csv")

df = df[["tweet", "BinaryNumTarget"]]
df = df.dropna()

df = df.sample(n=10000, random_state=42)

print("Dataset shape:", df.shape)

texts = df["tweet"].astype(str)
labels = df["BinaryNumTarget"].astype(int)

print("Loading GloVe...")
glove = load("glove-wiki-gigaword-100")


def text_to_vec(text):
    words = text.split()
    vecs = []

    for word in words:
        word = word.lower()
        if word in glove:
            vecs.append(glove[word])

    if len(vecs) == 0:
        return np.zeros(100)

    return np.mean(vecs, axis=0)


X = np.array([text_to_vec(t) for t in texts])
y = labels.values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="binary"
)

cm = confusion_matrix(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nROC-AUC:", roc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("GloVe + Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["False", "True"])
plt.yticks([0, 1], ["False", "True"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("glove_decision_tree_confusion_matrix.png", dpi=300)
plt.show()