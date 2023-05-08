'''
Adam Forestier
May 8, 2023
'''
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer, # Takes in count vector and terms into TF-IDF vector
    TfidfVectorizer
)

text = [
    'this is a line', 
    'this is another line right here',
    'completely different line that is pretty cool'
    ]

'''
Count Vectorizer: Count words and create Count Vectorizer
NOTE: stop_words removes common words. Either list you pass yourself or "english" for common english words
'''
cv = CountVectorizer(stop_words='english')

'''
# Get unique vocabularly and transform it to vector. Returns sparse matrix
# Will treat each item in the list as a seperate document
'''
sparse_matrix = cv.fit_transform(text) 

# .todense() NOTE: IMPORTANT - only do on tiny matrixes. Otherwise will each computer's memory. Just allows you to view matrix
# NOTE: Can do all functionality w/ sparse matrix!
dense_mat = sparse_matrix.todense()
# print(dense_mat) 


# Create tfidf from sparse matrix
tfidf = TfidfTransformer()
tfidf = tfidf.fit_transform(sparse_matrix) # Pass in bag of words -> transform to TF-IDF
print(tfidf.todense()) # REMINDER: Don't use to dense unless tiny amount of words. Just done here to visualize


# NOTE: DO IT ALL IN ONE STEP WITH TfIDF vectorizer
tv = TfidfVectorizer()
tv_results = tv.fit_transform(text)
print(tv_results.todense())