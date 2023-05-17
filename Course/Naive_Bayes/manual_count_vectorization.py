'''
Adam Forestier
May 7, 2023
'''

import pandas as pd

with open('One.txt') as my_txt:
    words_one = sorted(my_txt.read().lower().split())
    unique_words_one = set(words_one)

with open('Two.txt') as my_txt:
    words_two = sorted(my_txt.read().lower().split())
    unique_words_two = set(words_two)

unique_words = sorted(unique_words_one.union(unique_words_two))

full_vocab = dict()
index = 0
for word in unique_words:
    full_vocab[word] = index
    index += 1

one_freq = [0]*len(full_vocab)
two_freq = [0]*len(full_vocab)

def add_count_frequency(full_vocab=full_vocab, words=words_one, frequency_count=one_freq):
    '''
    arguments: vocab dictionary, words from txt document, frequency list
    returns: updated frequncy count list
    description: searches for word in dictionary, if word exists, update frequency count for that word by 1
    '''
    for word in words:
        word_index = full_vocab[word]
        if word_index:
            frequency_count[word_index] += 1
    return frequency_count

one_freq = add_count_frequency()
two_freq = add_count_frequency(full_vocab=full_vocab, words=words_two, frequency_count=two_freq)

bag_of_words = pd.DataFrame(data=[one_freq, two_freq], columns=full_vocab.keys())
print(bag_of_words)