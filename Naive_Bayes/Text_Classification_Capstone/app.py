'''
Adam Forestier
May 8, 2023
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df = pd.read_csv('moviereviews.csv')

# TASK: Check to see if there are any missing values in the dataframe.
null_count = df.isna().sum()

# TASK: Remove any reviews that are NaN
df = df.dropna()

# TASK: Check to see if any reviews are blank strings and not just NaN. Note: This means a review text could just be: "" or " " or some other larger blank string.
df = df[df['review'].str.strip().astype(bool)]

# TASK: Confirm the value counts per label:
vc = df['label'].value_counts()

# Bonus Task: Can you figure out how to use a CountVectorizer model to get the top 20 words (that are not english stop words) per label type? NOTE: COOL! Use in the future
count_vectorizer = CountVectorizer(stop_words='english')
pos = df[df['label'] ==  'pos']
neg = df[df['label'] ==  'neg']
pos_sparse_matrix = count_vectorizer.fit_transform(pos['review'])
neg_sparse_matrix = count_vectorizer.fit_transform(neg['review'])
pos_frequency = zip(count_vectorizer.get_feature_names_out(), pos_sparse_matrix.sum(axis=0).tolist()[0])
neg_frequency = zip(count_vectorizer.get_feature_names_out(), neg_sparse_matrix.sum(axis=0).tolist()[0])
sorted_pos = sorted(pos_frequency, key=lambda x: -x[1])[:20] # top 20 negative
sorted_neg = sorted(neg_frequency, key=lambda x: -x[1])[:20] # top 20 positive

# TASK: Split the data into features and a label (X and y) and then preform a train/test split. You may use whatever settings you like. To compare your results to the solution notebook, use test_size=0.20, random_state=101
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=.2, random_state=101)

# TASK: Create a PipeLine that will both create a TF-IDF Vector out of the raw text data and fit a supervised learning model of your choice. Then fit that pipeline on the training data
pipe = Pipeline([
    ('tfifd', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# TASK: Create a classification report and plot a confusion matrix based on the results of your PipeLine.
cr = classification_report(y_true=y_test, y_pred=y_pred)
print(cr)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
p.plot()
plt.show()