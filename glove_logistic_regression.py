import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

from gensim.downloader import load


df = pd.read_csv("dataset/Truth_Seeker_Model_Dataset.csv")

df = df[['tweet', 'BinaryNumTarget']]
df = df.dropna()

df = df.sample(n=10000, random_state=42)

print("Dataset shape:", df.shape)

texts = df["tweet"].astype(str)
labels = df["BinaryNumTarget"].astype(int)

print("Loading GloVe...")
glove = load("glove-wiki-gigaword-100")

def text_to_vec(text):
    words = text.split()
    vecs = [glove[w] for w in words if w in glove]
    if len(vecs) == 0:
        return np.zeros(100)
    return np.mean(vecs, axis=0)

X = np.array([text_to_vec(t) for t in texts])
y = labels.values


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


acc = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
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