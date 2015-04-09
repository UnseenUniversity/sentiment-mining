__author__ = 'alexei'


import pandas as pd
from bs4 import BeautifulSoup as bs
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer as Counter

import nltk
import sys


def get_file(file_name):

    if file_name == "labeled":
        return "../data/labeledTrainData.tsv"

    if file_name == "test":
        return "../data/testData.tsv"


def get_data(data_type):

    data = pd.read_csv(get_file(data_type), header=0,
                       delimiter="\t", quoting=3)
    return data

def filter_data(data):

    #filter tags/markup
    result = bs(data).get_text()

    #lower case
    result = result.lower()

    #TODO: remove punctuation
    result = re.sub("[^a-z.!?;:,]", " ", result)

    #split to tokens
    result = result.split()

    #remove stopwords
    #TODO: stemming
    result = [w for w in result if len(w) >= 2 and not w in stopwords.words("english")]

    #convert to string
    result = " ".join(result)

    return result



# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            sentences.append(filter_data(raw_sentence))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def tb_train():

    data = get_data("labeled")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    bigram_measures = nltk.collocations.BigramAssocMeasures()

    text = ""
    idx  = 0
    for review in data["review"]:
        text += filter_data(review)

        if idx == 2000:
            break

        idx += 1

    print "Initialize Counter.."

    tokens = nltk.wordpunct_tokenize(text)
    finder = nltk.BigramCollocationFinder.from_words(tokens, window_size=20)
    finder.apply_freq_filter(2)
    ignored_words = nltk.corpus.stopwords.words('english')
    finder.apply_word_filter(lambda w: len(w) < 3
                                       or '.' in w or '?' in w or '!' in w
                                       or w is "movie" or w is "film")

    print finder.nbest(bigram_measures.likelihood_ratio, 20)


    # sentences = []
    # for review in data["review"]:
    #     sentences += review_to_sentences(review, tokenizer)
    #     print sentences
    #     sys.exit(0)



def clean_data(data):

    num_reviews = data["review"].size
    clean_reviews = []

    for i in xrange(0, num_reviews):

        if (i + 1) % 1000 == 0:
            print "Review %d of %d\n" % (i, num_reviews)

        clean_reviews.append(filter_data(data["review"][i]))

    return clean_reviews


def bow_train(vectorizer):

    data = pd.read_csv(get_file("labeled"), header=0,
                       delimiter="\t", quoting=3)

    clean_reviews = clean_data(data)

    train_features = vectorizer.fit_transform(clean_reviews)
    train_features = train_features.toarray()

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_features, data["sentiment"])

    return forest


def dump_result(data, result):
    output = pd.DataFrame(data={"id": data["id"], "sentiment": result})
    output.to_csv("bow.csv", index=False, quoting=3)

# Bag of words
def bow():
    vectorizer = Counter(analyzer="word",
                         tokenizer=None,
                         preprocessor=None,
                         stop_words=None,
                         max_features=5000)

    model = bow_train(vectorizer)

    test = pd.read_csv(get_file("test"), header=0,
                       delimiter="\t", quoting=3)

    clean_tests = clean_data(test)
    test_features = vectorizer.transform(clean_tests)
    test_features = test_features.toarray()

    result = model.predict(test_features)

    dump_result(test, result)


def tutorial():

    tb_train()







def main():
    tutorial()

if __name__ == "__main__":
    main()