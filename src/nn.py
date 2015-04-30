__author__ = 'alexei'

import gensim
from data_processing.mongo import MongoORM
from data_processing.util import Timer
from pprint import pprint as pp

import nltk
from nltk.stem import WordNetLemmatizer as wnl
wnl = wnl()

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
import numpy as np

splitters = {"but", "however", "although", "though", "moreover"}

ignore_pos_set = {"c", "e", "p", "t", "w"}

ignore_word_set = {"of", "a", "an", "it", "the", "that", "this", "these", "those", "\\"}

def w2vec_model():

    model = gensim.models.Word2Vec.load('../data/w2v_for_rotten.model')
    model.init_sims(replace=True)
    return model


def miner(text, deps, features, negation):
    #, feature_vectors, model):

    # pp(text)

    sentences = []
    # vectors   = []

    current_sentence = []
    # current_vec = None

    negate = False
    for dep in deps:
        if dep[0] == 'neg':
            negate = True

    negation.append(negate)

    for j in xrange(len(text)):
        word = text[j]

        if word in ignore_word_set:
            continue

        if word in splitters:
            if len(current_sentence) > 0:
                sentences.append(current_sentence)
                # vectors.append(current_vec)

            # current_vec      = None
            current_sentence = []
            continue

        pos = nltk.pos_tag([word])[0][1].lower()

        if pos[0] in ignore_pos_set:
            continue

        # if word in model:
        #     if current_vec is None:
        #         current_vec = model[word]
        #     else:
        #         current_vec += model[word]

        current_sentence.append(word)

    if len(current_sentence) > 0:
        sentences.append(current_sentence)
        # vectors.append(current_vec)

    if len(sentences) > 1:
        features.append(sentences)
        # feature_vectors.append(current_vec)
    else:
        features += sentences
        # feature_vectors += vectors

def augument_dataset_with_negation():

    t = Timer()
    mongo = MongoORM("rotten")
    phrases = mongo.get_collection("phrases")

    for phrase in phrases:

        negation = []

        idx = 0
        while idx < len(phrase["deps"]):
            deps = phrase["deps"][idx]

            neg = False
            for dep in deps:
                if dep[0] == 'neg':
                    neg = True

            if neg:
                negation.append(True)
            else:
                negation.append(False)
            idx += 1

        mongo.update_item("training_set", {"_id": phrase["_id"]}, {"$set": {"neg": negation}})

    t.measure("rotten corpus augumented in: ")



def build_rotten_dataset():

    t = Timer()
    mongo = MongoORM("rotten")
    phrases = mongo.get_collection("phrases")

    for phrase in phrases:
        sentiment = phrase["sentiment"]

        features = []
        negation = []

        idx = 0
        while idx < len(phrase["text"]):
            miner(phrase["text"][idx], phrase["deps"][idx], features, negation)
            idx += 1

        if len(features) > 0:
            item = {"_id": phrase["_id"],
                    "features": features,
                    "neg": negation,
                    "label": sentiment}
            mongo.upsert_item("training_set", item)

    t.measure("rotten corpus built in: ")

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

def make_vector(model, feature_list):

    vector = np.zeros(100, dtype="float32")
    for feature in feature_list:
        if feature in model:
            vector += model[feature]
        else:
            # backoff to naive solution
            return None

    return vector

def build_training_set():

    model = w2vec_model()

    t = Timer()
    mongo = MongoORM("rotten")
    tset = mongo.get_collection("training_set")

    ds = SupervisedDataSet(100, 1)
    count = 0

    for train_set in tset:

        features = train_set["features"]
        label    = int(train_set["label"])

        vectors = []
        if len(features) > 1:
            continue

        for feature in features:

            if type(feature[0]) is not list:
                vector = make_vector(model, feature)
                if vector is None:
                    continue

                vectors.append(vector)
                ds.addSample(tuple(list(vector)), (label,))
                count += 1
            else:
                continue
                # if len(feature) > 2 or type(feature[0][0]) is list:
                #     print features
                #     exit(0)
                #
                # vectors.append(make_vector(model, feature[0]))
                # vectors.append(make_vector(model, feature[1]))

    t.measure("rotten training set built in: ")

    pp("Dataset size for single phrases: " + str(count))

    t = Timer()
    net = buildNetwork(100, 100, 1, bias=True) #, hiddenclass=TanhLayer)
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=1000000, continueEpochs=10)
    t.measure("network train for single phrases: ")

    NetworkWriter.writeToFile(net, '../data/single_phrase_nn.xml')

    # print net.activateOnDataset(ds)


def curiosity_query():

    t = Timer()
    mongo_rotten = MongoORM("rotten")
    mongo_words  = MongoORM("words")
    tset = mongo_rotten.get_collection("training_set")

    count = 0
    ds = SupervisedDataSet(6, 1)

    negs = {"n't", "not", "nor"}

    for train_set in tset:

        features = train_set["features"]

        if len(features) > 1:
            continue

        if type(features[0][0]) is list:
            continue

        score = [0, 0, 0, 0, 0]

        if len(train_set["neg"]) != 0:
            negation = train_set["neg"][0]
            if not negation:
                negation = 0
            else:
                negation = 1
        else:
            negation = 0

        for feature in features[0]:

            if feature in negs:
                negation = 1

            positive = mongo_words.get_item_by_id("positive", feature)
            if positive:
                score[positive["label"]] += 1
                continue

            negative = mongo_words.get_item_by_id("negative", feature)
            if negative:
                score[negative["label"]] += 1
                continue

            score[2] += 1

        score = [negation] + score
        ds.addSample(tuple(score), (int(train_set["label"]),))

        positive_count = float(score[5]) + float(0.5 * score[4])
        negative_count = float(score[1]) + float(0.5 * score[2])

        if negative_count == 0 and positive_count == 0:
            verdict = 2
        elif negative_count > 0 and not negation:
            if negative_count > 1.0:
                verdict = 0
            else:
                verdict = 1
        elif negative_count > 0 and negation:
            verdict = 1
        elif positive_count > 0 and negation:
            verdict = 1
        elif positive_count > 2:
            verdict = 4
        else:
            verdict = 3


        if abs(verdict - train_set["label"]) > 1:
        #         # if count > 10:
            pp((positive_count, negative_count))
            pp((score, "verdict", verdict, "label", train_set["label"], features))
            print

            count += 1
            if count == 10:
                exit(0)

        # positive_count = float(score[4]) + float(0.5 * score[3])
        # negative_count = float(score[0]) + float(0.5 * score[1])
        # negative_count *= 2.5
        #
        # verdict = 0
        # if positive_count > 0 and negative_count > 0:
        #     if positive_count - negative_count > 0:
        #         verdict = 3
        #     elif positive_count - negative_count < 0:
        #         verdict = 1
        #     else:
        #         verdict = 2
        # else:
        #     if positive_count >= 1.0:
        #         verdict = 4
        #     elif positive_count > 0:
        #         verdict = 3
        #     elif negative_count >= 1.0:
        #         verdict = 0
        #     elif negative_count > 0:
        #         verdict = 1
        #     else:
        #         verdict = 2
        #
        # if score[4] > 2 or score[0] > 2:
        #     if abs(verdict - train_set["label"]) > 1:
        #         # if count > 10:
        #         pp((positive_count, negative_count))
        #         pp((score, "verdict", verdict, "label", train_set["label"], features))
        #         print
        #
        #         count += 1
        #         if count == 20:
        #             exit(0)

            # exit(0)

    t.measure("curiosity satisfied in: ")
    # ds.saveToFile("greedy_data.xml")
    #
    # print "sents with no deps: ", count
    #
    # print len(ds)
    # net = buildNetwork(6, 1, 1, bias=True)
    # trainer = BackpropTrainer(net, ds)
    # trainer.trainUntilConvergence(verbose=True, validationProportion=0.20, maxEpochs=10000, continueEpochs=10)
    #
    # t.measure("greedy algo: ")
    # NetworkWriter.writeToFile(net, '../data/greedy.xml')


def main():
    # augument_dataset_with_negation()
    # build_rotten_dataset()
    # build_training_set()
    curiosity_query()


if __name__ == "__main__":
    main()