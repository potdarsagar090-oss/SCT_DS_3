import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from zipfile import ZipFile
from io import BytesIO
url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
response = requests.get(url)
outer_zip = ZipFile(BytesIO(response.content))
inner_zip_name = "bank-additional.zip"
with outer_zip.open(inner_zip_name) as inner_zip_file_data:
    inner_zip = ZipFile(BytesIO(inner_zip_file_data.read()))
csv_file_name = "bank-additional/bank-additional-full.csv"
with inner_zip.open(csv_file_name) as f:
    df = pd.read_csv(f, sep=";")
print("✅ Dataset Shape:", df.shape)
print(df.head())
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("y_yes", axis=1)
y = df_encoded["y_yes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No","Yes"], yticklabels=["No","Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.show()
