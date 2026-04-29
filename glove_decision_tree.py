import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.downloader import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


def text_to_vec(text, glove_model):
    vecs = []

    for word in text.split():
        if word in glove_model:
            vecs.append(glove_model[word])

    if not vecs:
        return np.zeros(100)

    return np.mean(vecs, axis=0)


def print_holdout_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
    )
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"True-class Precision: {precision:.4f}")
    print(f"True-class Recall: {recall:.4f}")
    print(f"True-class F1 Score: {f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"ROC-AUC (true class): {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake (0)", "True (1)"], digits=4))
    print("Confusion Matrix:")
    print(cm)

    return cm


df = pd.read_csv("dataset/Truth_Seeker_Model_Dataset.csv")
df = df[["tweet", "BinaryNumTarget"]].dropna()
df["BinaryNumTarget"] = df["BinaryNumTarget"].astype(int)
df["clean_tweet"] = df["tweet"].astype(str).apply(clean_text)
df = df.sample(n=10000, random_state=42)

print("Dataset shape:", df.shape)
print(df[["tweet", "clean_tweet", "BinaryNumTarget"]].head())
print("Class distribution:")
print(df["BinaryNumTarget"].value_counts())

print("Loading GloVe...")
glove = load("glove-wiki-gigaword-100")

X = np.array([text_to_vec(text, glove) for text in df["clean_tweet"]])
y = df["BinaryNumTarget"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {
    "accuracy": [],
    "true_precision": [],
    "true_recall": [],
    "true_f1": [],
    "roc_auc_true_class": [],
}

for train_idx, val_idx in cv.split(X, y):
    X_cv_train, X_cv_val = X[train_idx], X[val_idx]
    y_cv_train, y_cv_val = y[train_idx], y[val_idx]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_cv_train, y_cv_train)

    val_pred = model.predict(X_cv_val)
    val_prob = model.predict_proba(X_cv_val)[:, 1]

    cv_metrics["accuracy"].append(accuracy_score(y_cv_val, val_pred))
    cv_metrics["true_precision"].append(precision_score(y_cv_val, val_pred, pos_label=1))
    cv_metrics["true_recall"].append(recall_score(y_cv_val, val_pred, pos_label=1))
    cv_metrics["true_f1"].append(f1_score(y_cv_val, val_pred, pos_label=1))
    cv_metrics["roc_auc_true_class"].append(roc_auc_score(y_cv_val, val_prob))

print("\n=== 5-Fold Cross-Validation Metrics ===")
for metric, scores in cv_metrics.items():
    print(f"CV {metric}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Hold-Out Test Metrics ===")
cm = print_holdout_metrics(y_test, y_pred, y_prob)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("GloVe + Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Fake (0)", "True (1)"])
plt.yticks([0, 1], ["Fake (0)", "True (1)"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("glove_decision_tree_confusion_matrix.png", dpi=300)
plt.show()