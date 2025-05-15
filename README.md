#Tokenization:

from nltk.tokenize import word_tokenize
df['tokens'] = df['text'].apply(word_tokenize)

# Stop Word Removal:

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])


#Handling Special Characters and Numbers:

import re
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

#Stemming (using nltk):

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
df['tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])


#Lemmatization (using spaCy):

import spacy
nlp = spacy.load('en_core_web_sm')
df['tokens'] = df['tokens'].apply(lambda x: [token.lemma_ for token in nlp(" ".join(x))])


#Handling Emojis and Emoticons:

import emoji
df['text'] = df['text'].apply(lambda x: emoji.demojize(x))
