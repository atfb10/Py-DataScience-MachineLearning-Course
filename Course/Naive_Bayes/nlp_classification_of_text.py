'''
Adam Forestier
May 8, 2023
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

df = pd.read_csv('airline_tweets.csv')

# Visualize
sns.countplot(data=df, x='airline_sentiment')
plt.show()
sns.countplot(data=df, x='negativereason')
plt.xticks(rotation=90)
plt.show()
sns.countplot(data=df, x='airline', hue='airline_sentiment')
plt.show()

# We only care about negative. Not specifically neutral or positive - make binary to improve performance
df['airline_sentiment'] = df['airline_sentiment'].map({'neutral': 'positive', 'negative': 'negative', 'positive': 'positive'})
sns.countplot(data=df, x='airline_sentiment')
plt.show()

X = df['text']
y = df['airline_sentiment']

# NOTE: IMPORTANT!!!! DO NOT DO COUNT VECTORIZATION ON WHOLE DATA SET! only due on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=101)

# NOTE: Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(X_train) #NOTE: ONLY fit training data
X_test = tfidf.transform(X_test)

# Models
nb = MultinomialNB()
nb.fit(X_train, y_train)
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
rbf = SVC()
rbf.fit(X_train, y_train)
linear_svm = LinearSVC()
linear_svm.fit(X_train, y_train)


def report(model, X_test=X_test, y_test=y_test) -> None:
    preds = model.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=preds))
    return

print('NB')
report(nb)
print('Logistic')
report(log)
print('Radial Basis SVM')
report(rbf)
print('Linear SVM')
report(linear_svm)

# Linear Support Vector Machine Does the best

# NOTE: Ready to deploy Make pipeline
pipe = Pipeline([('tfidf', TfidfVectorizer()), 
                 ('svc', LinearSVC())])

# fit on all data
pipe.fit(X, y)

# Test final model
test_pred = pipe.predict(['good flight', 'horrible service'])
print(test_pred) # Predicts Positive for good flight and negative for horrible service!