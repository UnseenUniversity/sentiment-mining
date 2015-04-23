__author__ = 'alexei'


# import pandas as pd
from nltk import data, corpus
from nltk.stem import WordNetLemmatizer as wnl
from operator import itemgetter
import math
import random
import re
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork
import sys



class Parser():

    def __init__(self):

        self.tokenizer = data.load('tokenizers/punkt/english.pickle')
        self.stopwords = corpus.stopwords.words("english")
        self.wnl       = wnl()

    def clean_phrase(self, phrase):


        #remove delimitators
        phrase = re.sub(r"\.|\\|\?|!|:|\"|\(|\)|-|&| / | - |,|_|;|\*", " ", phrase)

        #split to tokens
        tokens = phrase.split()

        #remove stopwords
        tokens = [tok for tok in tokens if not ((tok.lower() in self.stopwords) or
                                                (len(tok) == 1) or
                                                (tok.isdigit() and int(tok) > 10))]

        result = []
        current_entity = None

        for token in tokens:

            if len(token) <= 1:
                continue

            if token.istitle():
                if current_entity:
                    current_entity += "_" + token.lower()
                else:
                    current_entity = token.lower()
            else:
                if current_entity:
                    result.append(current_entity)
                    current_entity = None

                token = self.wnl.lemmatize(token)
                result.append(token.lower())

        if current_entity:
            result.append(current_entity)

        return result

    def extract_sentences(self, text):

        sentences = self.tokenizer.tokenize(text)
        sentences = filter(lambda x: len(x) > 0, sentences)
        sentences = map(self.clean_phrase, sentences)

        return sentences

    def extract_terms(self, raw_text):

        # remove tags/markup
        raw_text = re.sub(r"<(.*?)>", "", raw_text.strip())
        sentences = self.extract_sentences(raw_text)

        if self.terms:

            for sentence in sentences:
                for item in sentence:
                    if item in self.terms:
                        self.terms[item] += 1
                    else:
                        self.terms[item] = 1
                    self.term_count += 1

        return sentences

    def extract_features(self, raw_text, feature_set):
        pass




def feature_counts_from_dataset(parser, dataset):

    t = Timer()
    parser.terms      = {}
    parser.term_count = 0

    data = extract_data(dataset)

    for review in data["review"]:
        parser.extract_terms(review)

    t.measure(dataset + " dataset parsed in: ")

    feature_set   = parser.terms
    feature_count = parser.term_count
    return feature_set, feature_count




def build_dataset(datatype):

    print "Build dataset for " + datatype + " data..."

    parser = Parser()

    inject_sentiment = datatype is "labeled"

    t = Timer()

    data = extract_data(datatype)

    num_reviews = len(data["review"])

    for idx in xrange(num_reviews):

        review = parser.extract_terms(data["review"][idx])

        if inject_sentiment:
            sentiment = data["sentiment"][idx]


    t.measure(datatype + " dataset built in: ")





def determine_features(feature_count):
    print "determine features..."

    p = Parser()
    feature_set1, feature_count1 = feature_counts_from_dataset(p, "labeled")
    feature_set2, feature_count2 = feature_counts_from_dataset(p, "unlabeled")

    # Bhattacharyya feature selection
    feature_set = []
    lg_count1   = math.log(feature_count1)
    lg_count2   = math.log(feature_count2)

    for (feature, count) in feature_set1.items():
        if count < 2:
            continue

        p1 = math.log(count) - lg_count1

        if feature in feature_set2:
            count = feature_set2[feature]
            if count < 2:
                continue

            p2 = math.log(count) - lg_count2
            feature_set.append((feature, p1 + p2))

    result = sorted(feature_set, key=itemgetter(1), reverse=True)[:feature_count]

    with open("../data/feature_set.feat", "wb") as outfile:
        for res in result:
            outfile.write("%s\n" % res[0])

    print "Feature selection done!"

class NeuralNet():

    def __init__(self, num_weights):

        self.num_weights = num_weights

        # init a random set of weights
        random.seed(42)
        self.weights = [random.random()] * self.num_weights





    def train_network(self):
        pass






def pow(x):
    return x ** 2

def main():

    #determine_features(8000)

    num_weghts = 3000


    trainingDataSet = buildDataSet("labeled")
    testDataSet     = buildDataSet("test")

    net = NeuralNet(num_weghts)





    # p = Proc()
    # t = Timer()
    # result = p.compute(pow, xrange(1024))
    # t.measure()

    #print result


if __name__ == "__main__":
    main()