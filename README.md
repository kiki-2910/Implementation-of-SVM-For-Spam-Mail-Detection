# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the spam email dataset.
2. Preprocess the data by selecting the message and label columns and converting labels (ham = 0, spam = 1).
3. Split the dataset into training and testing data.
4. Convert the text messages into numerical values using TF-IDF and train the Support Vector Machine model.
5. Predict whether messages are spam or ham and display the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S.KEERTHANA
RegisterNumber:  25004216
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("C:/Users/acer/Downloads/spam.csv", encoding='latin-1')

df = df.iloc[:, :2]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df = df.dropna()
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vectorized, y_train)

y_pred = svm_model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won a FREE cruise to the Andaman! Call now to claim your prize!",
    "Can you pick up some milk on your way home?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately."
]

print("\n" + "="*50)
print("TESTING WITH EXAMPLE MESSAGES:")
print("="*50)

for msg in test_messages:
    msg_vectorized = vectorizer.transform([msg])
    prediction = svm_model.predict(msg_vectorized)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: {msg[:50]}... -> {result}")
```

## Output:
<img width="770" height="747" alt="image" src="https://github.com/user-attachments/assets/33e1daf8-83c4-4837-ada3-f07c237a56d8" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
