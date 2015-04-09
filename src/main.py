__author__ = 'alexei'

from nltk import data, corpus
from nltk.stem import WordNetLemmatizer as wnl
from gensim import corpora, models, similarities
import gensim

from util import *
import re
import datagen

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
        # tokens = [tok for tok in tokens if not ((tok.lower() in self.stopwords) or
        #                                         (len(tok) == 1) or
        #                                         (tok.isdigit() and int(tok) > 10))]

        tokens = [tok for tok in tokens if not ((len(tok) == 1) or
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
        return sentences











def build_dataset(parser, datatype, inject=False):

    t = Timer()

    data = extract_data(datatype)
    num_reviews = len(data["review"])

    reviews   = []
    for idx in xrange(num_reviews):

        sentences = parser.extract_terms(data["review"][idx])
        reviews += sentences

    t.measure(datatype + " dataset built in: ")

    return reviews






def build_corpus():

    p = Parser()
    set1 = build_dataset(p, "labeled", inject=True)
    set2 = build_dataset(p, "unlabeled")
    set1 += set2

    # dictionary = corpora.Dictionary(set1)
    # dictionary.save('../data/dataset.dict')
    #
    # print dictionary

    return set1


def word2vec():

    sentences = build_corpus()

    t = Timer()
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    t.measure("word2vec model built in: ")

    model.save('../data/word2vec_nostop.model')

def get_feature_cluster(model, seed_list):

    seed_set = set(seed_list)
    features = set()


    while len(seed_list) > 0:

        seed = seed_list.pop(0)
        features.add(seed)

        # print seed

        try:
            sim_words = model.most_similar_cosmul(positive=list(features), topn=20)
        except KeyError:
            return features

        for feat in sim_words:

            word, score = feat

            if score < 0.65:
                break

            if not "_" in word:
                if word not in seed_set:
                    seed_list.append(word)
                    seed_set.add(word)

    return features


def word2vec_fun():

    model = gensim.models.Word2Vec.load('../data/word2vec_nostop.model')
    model.init_sims(replace=True)
    # print model.most_similar_cosmul(positive=['begin'], topn=20)
    # print nltk.pos_tag(["completely", "undeserved", "prize"])

    # for actor in good_actors:
    # print model.most_similar_cosmul(positive=['not', 'amazing', 'actor'], topn=10)
    # print model.most_similar_cosmul(positive=['actress'], negative=["man"], topn=10)
    # print model.similarity('enthralls', 'great')
    # print model.similarity('NEG', 'great')

    return model


import nltk

def spawn_feature_vectors():

    model = word2vec_fun()

    with open("../data/feature_vectors.csv", "a") as outfile:
        feature_vector = get_feature_cluster(model, ["insight"])

        string = ""
        for feature in feature_vector:
            string += "\t" + feature
        outfile.write("%s\n" % string)


def main():

    # build_corpus()
    # word2vec()
    # word2vec_fun()
    # test()
    # spawn_feature_vectors()
    feature_builder()

def test():

    model = word2vec_fun()

    p = Parser()
    ref = datagen.reference_labels()

    data = extract_data("test")
    num_reviews = len(data["review"])

    reviews   = []
    for idx in xrange(num_reviews):

        sentences = p.extract_terms(data["review"][idx])

        print "label", ref[data["id"][idx].replace("\"", "")]

        if idx == 1:
            for sentence in sentences:
                print sentence
            exit(0)
            # print get_feature_cluster(model, sentence)
            # exit(0)



        #print sentences
        # exit(0)


def feature_extraction():

    feature_vectors = {}
    feature_map     = {}

    path = "../data/feature_vectors.csv"
    for idx, line in enumerate(open(path, "rb")):

        feats = set(line.split())
        feature_vectors[idx] = feats
        for elem in feats:
            if elem in feature_map:
                print elem
                print feature_vectors[feature_map[elem]]
                print feats
                exit(0)

            feature_map[elem] = idx

    return feature_vectors, feature_map

def update_dict(fdict, word, label):

    if word in fdict:
        (neg, pos) = fdict[word]
    else:
        (neg, pos) = (0, 0)

    if label:
        fdict[word] = (neg, pos + 1)
    else:
        fdict[word] = (neg + 1, pos)

import math

def feature_builder():

    feature_dict = {}


    feature_vectors, feature_map = feature_extraction()
    num_vectors = len(feature_vectors) + 1
    feature_vectors[num_vectors] = set()



    print "number of feature vectors", num_vectors

    model = word2vec_fun()
    parser = Parser()

    data = extract_data("labeled")
    num_reviews = len(data["review"])

    samples = []
    sample_count = [0, 0]

    for idx in xrange(num_reviews):

        sentences = parser.extract_terms(data["review"][idx])
        label = int(data["sentiment"][idx])
        sample_count[label] += 1

        for sentence in sentences:

            prior = -1
            sample = []

            for word in sentence:

                if word in feature_map and feature_map[word] != num_vectors:
                    sample.append((feature_map[word], word))
                    prior = feature_map[word]
                else:

                    best_score = 0.0
                    best_id    = 0

                    # for jdx in feature_vectors:
                    #     vec = feature_vectors[jdx]
                    #     score = 0.0
                    #     for elem in vec:
                    #         score += model.similarity(word, elem)
                    #     score /= len(vec)
                    #     if score > best_score:
                    #         best_score, best_id = score, jdx
                    #
                    # print best_score, best_id
                    #
                    # if best_score > 0.6:
                    #     sample.append((best_id, word))
                    #     feature_map[word] = best_id
                    # else:
                    if prior == num_vectors:
                        (feat_id, gram) = sample[len(sample) - 1]
                        sample[len(sample) - 1] = (feat_id, gram + "_" + word)
                    else:
                        sample.append((num_vectors, word))
                        feature_map[word] = num_vectors
                        feature_vectors[num_vectors].add(word)
                        prior = num_vectors

            # for (_, w) in sample:
            #     update_dict(feature_dict, w, label)

            if len(sample) > 1:
                samples.append((sample, label))

        # print samples
        # exit(0)
        # break

    # with open("../data/content_structure.csv", "a") as outfile:
    #     for sample in samples:
    #         line = ""
    #         for feature in sample:
    #             line += "\t" + str(feature[0])
    #         outfile.write("%s\n" % line)

    with open("../data/feature_sequence.csv", "wb") as outfile:

        result_dict = {}

        for sample in samples:

            feat_list = []
            label     = sample[1]

            for feature in sample[0]:

                feat_list.append(feature[0])
                _len = len(feat_list)

                if _len > 5:
                    feat_list.pop(0)

                if _len >= 4:
                    v = (feat_list[-4], feat_list[-3], feat_list[-2], feat_list[-1])
                    update_dict(result_dict, v, label)

                if _len >= 3:
                    v = (feat_list[-3], feat_list[-2], feat_list[-1])
                    update_dict(result_dict, v, label)

                if _len >= 2:
                    v = (feat_list[-2], feat_list[-1])
                    update_dict(result_dict, v, label)

        outfile.write("\t%s\t%s\t\n" % (sample_count[0], sample_count[1]))

        for value in result_dict:
            (neg, pos) = result_dict[value]
            outfile.write("%s\t%s\t%s\t%s\t\n" % (value, neg, pos, pos - neg))

    #
    #     total_count = sample_count[0] + sample_count[1]
    #     sample_count[0] /= total_count
    #     sample_count[1] /= total_count
    #
    #     print sample_count
    #
    #     for feature in feature_dict:
    #         (neg, pos) = feature_dict[feature]
    #         if neg + pos >= 5:
    #             outfile.write("%s\t%s\t%s\n" % (feature, neg, pos))


    # # t = Timer()
    # for idx in xrange(num_reviews):
    #
    #     # if idx % 10 == 0:
    #     print (idx + 1), len(feature_vectors), "..."
    #
    #     sentences = parser.extract_terms(data["review"][idx])
    #
    #     for sentence in sentences:
    #         for word in sentence:
    #             if word not in feature_map:
    #                 vector = get_feature_cluster(model, [word])
    #
    #                 if len(vector) == 1:
    #                     continue
    #
    #                 jdx = len(feature_vectors) + 1
    #                 feature_map[word] = jdx
    #                 feature_vectors[jdx] = vector
    #                 feature_count[jdx] = 1
    #             else:
    #                 feature_count[feature_map[word]] += 1
    #
    #         print feature_vectors
    #         print feature_count
    #         exit(0)

        # reviews += sentences

    # with open("../data/feature_vectors.csv", "wb") as outfile:
    #     for idx in feature_count:
    #         outfile.write("%s\t%s\n" % (feature_count[idx], feature_vectors[idx]))

        # t.measure("labeled" + " dataset built in: ")

    # return reviews



if __name__ == "__main__":
    main()