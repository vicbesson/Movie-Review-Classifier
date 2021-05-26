import collections
import nltk
from nltk.metrics.scores import (precision, recall)
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from sklearn import model_selection

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:5000]
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set,test_set = model_selection.train_test_split(featuresets,test_size = 0.30)
classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print('Classifier Accuracy:', nltk.classify.accuracy(classifier, test_set))
print('pos precision:', nltk.precision(refsets['pos'], testsets['pos']))
print('pos recall:', nltk.recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', nltk.f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', nltk.precision(refsets['neg'], testsets['neg']))
print('neg recall:', nltk.recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', nltk.f_measure(refsets['neg'], testsets['neg']))
classifier.show_most_informative_features(10)
print('Enter movie review to classify:')
user_review = input()
user_review_features = {word: (word in word_tokenize(user_review.lower())) for word in all_words}
print('Your review was classified as',classifier.classify(user_review_features))
user_review = input()
