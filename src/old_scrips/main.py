__author__ = 'alexei'


from util import *
import gensim
import datagen

import math
from gensim import corpora, models, similarities

import operator

# from sent2vec.word2vec import Sent2Vec

def train_paragraph_vectors():

    p = Parser()
    set = build_dataset(p, "labeled", inject=True)

    print set[:2]

    exit(0)
    model = Sent2Vec(set, model_file='../data/word2vec_augumented.model')
    model.save_sent2vec_format("../paragraph_model.vec")





def build_corpus():

    p = Parser()
    ans = []


    set0 = build_dataset(p, "rotten")
    set1 = build_dataset(p, "labeled", inject=True)
    set2 = build_dataset(p, "unlabeled")
    set3 = build_dataset(p, "test")

    ans += set0
    ans += set1
    ans += set2
    ans += set3

    counter = {}
    res = set()
    for phrase in ans:
        for word in phrase:
            if word in res:
                pass
            else:
                if word in counter:
                    counter[word] += 1
                    if counter[word] > 10:
                        res.add(word)
                else:
                    counter[word] = 1

    dictionary = corpora.Dictionary(list(res))
    dictionary.save('../data/full_dataset.dict')
    print dictionary

    return ans

def word2vec_train():

    sentences = build_corpus()

    print "TrainWord2Vec: ", len(sentences)

    t = Timer()
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    t.measure("word2vec model built in: ")
    model.save('../data/word2vec_augumented.model')

    return model

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

            if score < 0.89:
                break

            if not "_" in word:
                if word not in seed_set:
                    seed_list.append(word)
                    seed_set.add(word)

    return features


def word2vec_fun():

    model = gensim.models.Word2Vec.load('../data/word2vec_augumented.model')
    model.init_sims(replace=True)
    # print model.most_similar_cosmul(positive=['begin'], topn=20)
    # print nltk.pos_tag(["completely", "undeserved", "prize"])

    # print model["great"]

    # for actor in good_actors:
    # print model.most_similar_cosmul(positive=['floundering'], topn=20)
    # print model.most_similar_cosmul(positive=['actress'], negative=["man"], topn=10)
    # print model.similarity('enthralls', 'great')
    # print model.similarity('NEG', 'great')
    return model

def cluster_wordvectors():

    model = word2vec_fun()
    dictionary = corpora.Dictionary.load('../data/full_dataset.dict')
    prior = load_words_prior()
    (clusters, word_cluster) = feature_extraction()
    new_words = set()

    for word in dictionary.token2id:

        if "_" in word or word in word_cluster:
            continue

        label = 0
        if word in prior:
            label = prior[word]

        best_score = 0
        best_idx = -1

        cluster = get_feature_cluster(model, [word])

        if len(cluster) == 1:
            new_words.add(word)
        else:
            for idx in clusters:

                other = clusters[idx]

                if label == 0 or label == other[0]:
                    features = other[1]

                    common = cluster & features
                    if len(common) >= 2 and len(common) > best_score:
                        best_score = len(common)
                        best_idx   = idx

            # print label, word
            # print cluster
            #
            # print best_score
            # print best_idx
            # print clusters[best_idx][1]

            if best_idx == -1:
                if len(cluster) > 1:
                    # print "new cluster", cluster
                    clusters[len(clusters)] = (label, cluster)
                    for w in cluster:
                        word_cluster[w] = len(clusters)
                else:
                    new_words.add(word)
            else:
                # print "append word ", word, "to cluster", best_idx
                clusters[best_idx][1].add(word)

    clusters[len(clusters)] = (0, new_words)

    with open("../data/full_clusters.csv", "w") as outfile:
        for idx in clusters:
            cluster = clusters[idx]
            outfile.write("%d" % int(cluster[0]))
            for word in cluster[1]:
                try:
                    outfile.write("\t%s" % word)
                except UnicodeEncodeError:
                    continue
            outfile.write("\n")


import nltk

def spawn_feature_vectors():

    model = word2vec_fun()

    with open("../data/feature_vectors.csv", "a") as outfile:
        feature_vector = get_feature_cluster(model, ["insight"])

        string = ""
        for feature in feature_vector:
            string += "\t" + feature
        outfile.write("%s\n" % string)


def feature_extraction(path):

    feature_vectors = {}
    feature_map     = {}

    path = "../data/" + path #new_feature_vectors.csv"
    for idx, line in enumerate(open(path, "rb")):

        elems = line.split()
        feats = set(elems[1:])
        feature_vectors[idx] = (elems[0], feats)
        for elem in feats:
            if elem in feature_map:
                print "Duplicate seed!!!"
                print elem, feats, feature_vectors[feature_map[elem]]
                exit(0)

            feature_map[elem] = idx

    return feature_vectors, feature_map


from spacy.en import English
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
import numpy as np

def nn_attempt():

    #(feature_vector, feature_map) = feature_extraction("full_clusters.csv")
    # data = extract_data("labeled")

    # parser = Parser()
    # num_reviews = len(data["review"])

    model = word2vec_fun()

    num_features = len(model["very"])

    ds = SupervisedDataSet(100, 1)

    # "actor" index is 1
    empty = np.zeros((num_features,), dtype="float32")

    positive = ["fantastic", "outstanding", "awesome", "magnificent",	"marvelous",	"fabulous",	"fine",	"impressive",	"quality",	"tremendous",	"phenomenal",
                "superb", "terrific", "amazing", "wonderful",	"amusing",	"pretty",	"nice",	"good",	"exceptional",	"excellent",	"talented",	"great",	"splendid",
                "brilliant", "neat", "interesting", "commendable",	"greatest",
                "finest", "best", "beautiful", "lovely",	"wonderfully",	"brilliantly",	"superbly",	"beautifully",	"gorgeous",	"excellently",	"capably",	"masterfully",
                "expertly", "brilliantly", "incredible"]

    positive2 = ["9/10", "5/5", "4/5", "8/10", "7/10", "10/10"]
    negative2 = ["1/10", "2/10", "3/10", "4/10", "5/10", "6/10"]
    prefix = np.add(empty, model["rating"])
    prefix = np.add(prefix, model["movie"])

    for w in positive2:
        ds.addSample(tuple(list(np.add(prefix, model[w]))), (1,))
    for w in positive2:
        ds.addSample(tuple(list(np.add(prefix, model[w]))), (1,))

    for w in negative2:
        ds.addSample(tuple(list(np.add(prefix, model[w]))), (0,))
    for w in negative2:
        ds.addSample(tuple(list(np.add(prefix, model[w]))), (0,))


    negative = []

    # not_vec = np.add(empty, model["not"])
    # idx = 0
    # for w in positive:
    #     ds.addSample(tuple(list(np.add(empty, model[w]))), (1,))
    #     ds.addSample(tuple(list(np.add(not_vec, model[w]))), (0,))

    # ds.addSample(tuple(list(np.add(features, model["good"]))), (1,))
    # ds.addSample(tuple(list(np.add(features, model["great"]))), (1,))
    # ds.addSample(tuple(list(np.add(features, model["amazing"]))), (1,))
    # ds.addSample(tuple(list(np.add(features, model["terrible"]))), (0,))
    # ds.addSample(tuple(list(np.add(features, model["terrible"]))), (0,))
    # ds.addSample(tuple(list(np.add(features, model["bad"]))), (0,))
    # ds.addSample(tuple(list(np.add(features, model["best"]))), (1,))
    # ds.addSample(tuple(list(np.add(t1, model["good"]))), (0,))
    # ds.addSample(tuple(list(np.add(t1, model["great"]))), (0,))
    # ds.addSample(tuple(list(np.add(t1, model["amazing"]))), (0,))
    # ds.addSample(tuple(list(np.add(t1, model["bad"]))), (1,))
    print len(ds)

    net = buildNetwork(100, 500, 1, bias=True)#, hiddenclass=TanhLayer)
    trainer = BackpropTrainer(net, ds)

    trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=100000, continueEpochs=10)

    ds2 = SupervisedDataSet(100, 1)
    ds2.addSample(tuple(list(np.add(prefix, model["7/10"]))), (1,))

    print net.activateOnDataset(ds)
    print net.activateOnDataset(ds2)



    # for idx in xrange(num_reviews):
    #     sentences = parser.extract_terms(data["review"][idx])
    #     label     = data["sentiment"][idx]
    #
    #
    #
    #
    #     print sentences, label
    #     exit(0)





def main():

    # build_corpus()
    # word2vec_train()

    # word2vec_fun()

    # print feature_extraction()
    # cluster_wordvectors()

    # test()
    # spawn_feature_vectors()
    # feature_builder()

    # nn_attempt()

    train_paragraph_vectors()

    pass

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


def update_dict(fdict, word, label):

    if word in fdict:
        (neg, pos) = fdict[word]
    else:
        (neg, pos) = (0, 0)

    if label:
        fdict[word] = (neg, pos + 1)
    else:
        fdict[word] = (neg + 1, pos)


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