import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'message': [
        'Congratulations! You won a free iPhone',
        'Hey, are you coming to college today?',
        'Claim your lottery prize now',
        'Can we meet after class?',
        'Get free recharge instantly',
        'Your account has been credited',
        'Win cash now!!!',
        'Happy birthday my friend',
        'Click this link to earn money fast',
        'Let us study together tomorrow'
    ],
    'label': [
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham'
    ]
}

df = pd.DataFrame(data)

X = df['message']
y = df['label']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

new_message = ["You won free cash prize"]

new_message_vectorized = vectorizer.transform(new_message)

result = model.predict(new_message_vectorized)

print("Message:", new_message[0])
print("Prediction:", result[0])