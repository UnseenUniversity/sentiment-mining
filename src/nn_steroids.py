__author__ = 'alexei'

from data_processing.mongo import MongoORM
from pprint import pprint as pp
from data_processing.util import *
import gensim


def extract_sentences(mongo, collection, result):

    items = mongo.get_collection(collection)
    for item in items:
        if len(item["sentences"]) > 1:
            result += item["sentences"]


def get_entire_corpus():

    mongo = MongoORM("corpus")

    t = Timer()
    result = []
    extract_sentences(mongo, "labeled", result)
    extract_sentences(mongo, "unlabeled", result)
    extract_sentences(mongo, "test", result)
    extract_sentences(mongo, "rotten_train", result)
    extract_sentences(mongo, "rotten_test", result)

    pp("count sentences: " + str(len(result)))
    t.measure("got all sentences: ")
    return result


def w2vec_train(sentences, dest=None):

    if dest is None:
        dest = '../data/w2v_corpus.model'

    print "Train w2vec model: "
    t = Timer()
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=4, workers=4)
    model.save(dest)
    t.measure("word2vec model built in: ")


def w2vec_model(path=None):

    if path is None:
        path = '../data/w2v_corpus.model'

    model = gensim.models.Word2Vec.load(path)
    model.init_sims(replace=True)
    return model


def train_word_vectors():
    result = get_entire_corpus()
    w2vec_train(result)
    pp("Train success!")

###############################################

import numpy as np
import nltk

word_dict = {}
ignore_word_set = {"of", "a", "an", "it", "the", "that", "this", "these", "those", "\\"}

def extract_sentence_vectors(sentence, model):

    global word_dict
    vector = np.zeros(100, dtype="float32")

    for word in sentence:

        if word in word_dict:
            if word_dict[word]:
                vector += model[word]
            continue

        if word in ignore_word_set or word not in model:
            word_dict[word] = False
            continue

        pos = nltk.pos_tag([word])[0][1][0].lower()
        if pos == 'e' or pos == "w" or pos == "t" or pos == "s":
            word_dict[word] = False
            continue

        word_dict[word] = True
        vector += model[word]

    return vector


def extract_phrase_vectors(sentences, model, result):

    global word_dict

    for sentence in sentences:

        if len(sentence) <= 1:
            continue

        vector = np.zeros(100, dtype="float32")

        for word in sentence:

            if word in word_dict:
                if word_dict[word]:
                    vector += model[word]
                continue

            if word in ignore_word_set or word not in model:
                word_dict[word] = False
                continue

            pos = nltk.pos_tag([word])[0][1][0].lower()
            if pos == 'e' or pos == "w" or pos == "t" or pos == "s":
                word_dict[word] = False
                continue

            word_dict[word] = True
            vector += model[word]

        if result.size == 0:
            result = np.array([vector])
        else:
            result = np.append(result, [vector], 0)

    return result

def extract_review_vectors(mongo, collection, model, result):

    t = Timer()

    counter = 1

    items = mongo.get_collection_large(collection)

    try:
        for item in items:

            result = extract_phrase_vectors(item["sentences"], model, result)
            if counter % 1000 == 0:
                pp(str(counter) + " items mined.")
            counter += 1

    except Exception:
        pass

    items.close()
    t.measure("Vectors from reviews extracted in:")
    return result

from sklearn.cluster import KMeans
import pickle

def cluster_phrases():

    mongo = MongoORM("corpus")
    model = w2vec_model()
    result = np.array([])
    result = extract_review_vectors(mongo, "labeled", model, result)

    estimator = KMeans(init='k-means++', n_clusters=30, n_init=10, n_jobs=-1, verbose=1)
    pp(result)

    t = Timer()
    estimator.fit(result)
    t.measure("clustering operation success in:")

    pickle.dump(estimator, open('../data/phrase_clusters_30.pickel', 'wb'))


def dump_test_results(clusters_pos, clusters_neg, clusters_counter):

    global word_dict
    pp("Reaady to rumble!")

    class_count = 30

    with open('../data/sentence_clusters_30.test_res', 'w') as f:

        f.write("Test results:\n")

        for _class in xrange(class_count):
            f.write("Count for class " + str(_class) + " : ")
            f.write(str(clusters_counter[0][_class]) + " / " + str(clusters_counter[1][_class]))
            f.write("\n")

        f.write("\n")
        f.write("\n")

        for idx in xrange(class_count):

            f.write("Class " + str(idx) + "\n")

            f.write("Positive:\n")
            for sent in clusters_pos[idx]:
                for token in sent:
                    if token in word_dict and\
                       word_dict[token]:
                        f.write(token + " ")
                    else:
                        f.write("(" + token + ") ")

                f.write("\n")

            f.write("\n")

            f.write("Negative:\n")
            for sent in clusters_neg[idx]:
                for token in sent:
                    if token in word_dict and\
                       word_dict[token]:
                        f.write(token + " ")
                    else:
                        f.write("(" + token + ") ")
                f.write("\n")

            f.write("\n")


def cluster_test():

    estimator = pickle.load(open('../data/phrase_clusters_30.pickel', 'rb'))
    model = w2vec_model()
    mongo = MongoORM("corpus")

    # item = mongo.get_item_by_id("labeled", "2381_9")
    # sents = item["sentences"]
    # sents = [["plot", "actor", "performance"], ["I", "rate", "this", "movie", "a", "10"], ["very", "beautiful", "movie"]]
    #
    # for sent in sents:
    #     vector = extract_sentence_vectors(sent, model)
    #     print estimator.predict([vector])

    clusters_pos = {}
    clusters_neg = {}
    clusters_counter = [{}, {}]
    class_count = 30

    for idx in xrange(class_count):
        clusters_pos[idx] = []
        clusters_neg[idx] = []
        clusters_counter[0][idx] = 0
        clusters_counter[1][idx] = 0

    rest_pos, rest_neg = class_count, class_count
    batch_size = 15

    t = Timer()
    for item in mongo.get_collection("labeled"):

        label = int(item["label"])

        for sentence in item["sentences"]:

            if len(sentence) <= 1:
                continue

            vector = extract_sentence_vectors(sentence, model)
            _class = estimator.predict([vector])[0]

            clusters_counter[label][_class] += 1

            if rest_pos == 0 and rest_neg == 0:
                continue

            if label:
                if len(clusters_pos[_class]) < batch_size:
                    clusters_pos[_class].append(sentence)
                    # pp("found pos " + str(_class))

                    if len(clusters_pos[_class]) == batch_size:
                        rest_pos -= 1
                        # if rest_pos == 0 and rest_neg == 0:
                        #     dump_test_results(clusters_pos, clusters_neg, clusters_counter)
                            # exit(0)
            else:
                if len(clusters_neg[_class]) < batch_size:
                    clusters_neg[_class].append(sentence)
                    # pp("found neg " + str(_class))

                    if len(clusters_neg[_class]) == batch_size:
                        rest_neg -= 1
                        # if rest_pos == 0 and rest_neg == 0:
                        #     dump_test_results(clusters_pos, clusters_neg, clusters_counter)
                            # exit(0)

    t.measure("vectors attributed to clusters!")

    pp("reached the end of the straw.")
    dump_test_results(clusters_pos, clusters_neg, clusters_counter)

def store_cluster_results():

    estimator_10 = pickle.load(open('../data/phrase_clusters.pickel', 'rb'))
    estimator_30 = pickle.load(open('../data/phrase_clusters_30.pickel', 'rb'))

    model = w2vec_model()
    mongo = MongoORM("corpus")
    mongo_dump = MongoORM("clusters")

    count = 0

    t = Timer()
    for item in mongo.get_collection("labeled"):

        label = int(item["label"])

        for sentence in item["sentences"]:

            if len(sentence) <= 1:
                continue

            vector = extract_sentence_vectors(sentence, model)
            vector = [elem.item() for elem in vector]

            _class_10 = estimator_10.predict([vector])[0]
            _class_30 = estimator_30.predict([vector])[0]

            item = {"sent": sentence,
                    "vector": vector,
                    "class_10": _class_10.item(),
                    "class_30": _class_30.item(),
                    "label": int(label)
                    }

            mongo_dump.insert_item("clustered_phrases", item)

        count += 1
        if count % 1000 == 0:
            pp(str(count) + " items mined.")

    t.measure("Cluster results successfully dumped in ")


from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader


def get_samples(mongo, cluster_idx, dataset, label, count_max):

    count = 0
    for item in mongo.get_item("clustered_phrases", {"class_30": cluster_idx, "label": label}):
        vector = item["vector"]
        dataset.addSample(tuple(vector), (label,))

        count += 1
        if count == count_max:
            break

def test_net(mongo_cursor, net, append=True):

    if append:
        mod = 'a'
    else:
        mod = 'w'

    with open('../data/debug.res', mod) as f:

        for item in mongo_cursor:
            f.write("Label: ")
            f.write(str(net.activate(item["vector"])))
            f.write(" || ")
            f.write(str(item["label"]))
            f.write("\n")

            for token in item["sent"]:
                f.write(token + " ")
            f.write("\n")
            f.write("\n")


def build_dataset_for_nn_training(cluster_idx):

    mongo = MongoORM("clusters")
    ds = SupervisedDataSet(100, 1)

    t = Timer()
    count_pos = mongo.get_item_count("clustered_phrases", {"class_30": cluster_idx, "label": 1})
    count_neg = mongo.get_item_count("clustered_phrases", {"class_30": cluster_idx, "label": 0})
    count_min = min(count_pos, count_neg)
    get_samples(mongo, cluster_idx, ds, 1, count_min)
    get_samples(mongo, cluster_idx, ds, 0, count_min)
    t.measure("Dataset built! ")
    pp(str(len(ds)) + " elements.")

    return ds


def train_nn_for_cluster(net, dataset, cluster_idx):
    t = Timer()
    trainer = BackpropTrainer(net, dataset)
    trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=20000, continueEpochs=10)
    t.measure("network train for single phrases: ")
    NetworkWriter.writeToFile(net, '../data/cluster_net' + str(cluster_idx) + ".xml")


def test_network(cluster_idx, net=None):

    mongo = MongoORM("clusters")

    if net is None:
        net = NetworkReader.readFrom('../data/cluster_net' + str(cluster_idx) + ".xml")

    cursor_neg = mongo.get_item("clustered_phrases", {"class_30": cluster_idx, "label": 1})
    test_net(cursor_neg, net, append=False)

    cursor_pos = mongo.get_item("clustered_phrases", {"class_30": cluster_idx, "label": 0})
    test_net(cursor_pos, net)

def train_network(cluster_idx):

    ds = build_dataset_for_nn_training(cluster_idx)
    net = buildNetwork(100, 80, 1, bias=True) #, hiddenclass=TanhLayer)
    train_nn_for_cluster(net, ds, cluster_idx)
    test_network(cluster_idx, net)


def retrain_network(cluster_idx):

    ds = build_dataset_for_nn_training(cluster_idx)
    net = NetworkReader.readFrom('../data/cluster_net' + str(cluster_idx) + ".xml")
    train_nn_for_cluster(net, ds, cluster_idx)
    test_network(cluster_idx, net)

############################################################

def w2vec_train_steroids():

    mongo = MongoORM("clusters")

    t = Timer()
    corpus = {0: [], 1: []}
    for item in mongo.get_collection("clustered_phrases"):
        label = item["label"]
        corpus[label].append(item["sent"])

    mongo = MongoORM("corpus")

    for item in mongo.get_collection("rotten_train"):
        label = item["label"]
        if label == 0:
            for sentence in item["sentences"]:
                corpus[label].append(sentence)
        elif label == 4:
            for sentence in item["sentences"]:
                corpus[1].append(sentence)

    t.measure("dataset built!")

    w2vec_train(corpus[0], '../data/w2v_0_corpus.model')
    w2vec_train(corpus[1], '../data/w2v_1_corpus.model')
    w2vec_test()

import scipy.spatial.distance as d

def sum_vectors(model, sentence):

    result = None
    for word in sentence:
        if result is None:
            result = model[word]
        else:
            result += model[word]

    return result


def score_word(model, word, prefix, sufix):

    pp("Prefix score: ")
    pp(d.cosine(prefix, model[word]))
    # print d.cosine(gensim.matutils.unitvec(prefix), gensim.matutils.unitvec(model[word]))

    pp("Sufix score:")
    pp(d.cosine(model[word], sufix))
    # print d.cosine(gensim.matutils.unitvec(sufix), gensim.matutils.unitvec(model[word]))



def w2vec_test():

    neg_model = w2vec_model('../data/w2v_0_corpus.model')
    pos_model = w2vec_model('../data/w2v_1_corpus.model')
    baseline  = w2vec_model('../data/w2v_corpus.model')

    # poor thrown together movie matrix reload be disappointment
    # sentence = ["This", "be", "very", "good", "movie"]

    print "Negative model "
    # prefix = sum_vectors(neg_model, ["poor"])
    # sufix  = sum_vectors(neg_model, ["matrix"])
    # word   = "movie"
    # score_word(neg_model, word, prefix, sufix)

    # pp(neg_model.most_similar_cosmul(positive=['movie'], topn=15))
    pp(neg_model.similarity("be", "good"))
    pp(neg_model.similarity("worth", "seek"))
    pp(neg_model.similarity("poor", "movie"))
    pp(neg_model.similarity("truly", "masterpiece"))


    # pp(neg_model.similarity("not", "very"))

    print
    print "Positive model "
    # prefix = sum_vectors(pos_model, ["poor"])
    # sufix  = sum_vectors(pos_model, ["matrix"])
    # word   = "movie"
    # score_word(pos_model, word, prefix, sufix)

    pp(pos_model.similarity("be", "good"))
    pp(pos_model.similarity("worth", "seek"))
    pp(pos_model.similarity("poor", "movie"))
    pp(pos_model.similarity("truly", "masterpiece"))

    # pp(pos_model.most_similar_cosmul(positive=['movie'], topn=15))
    # pp(pos_model.similarity("not", "very"))

    print
    print "Baseline model "
    # prefix = sum_vectors(baseline, ["poor", "thrown", "together"])
    # sufix  = sum_vectors(baseline, ["matrix"])
    # word   = "movie"
    # score_word(baseline, word, prefix, sufix)

    pp(baseline.similarity("be", "good"))
    pp(baseline.similarity("worth", "seek"))
    pp(baseline.similarity("poor", "movie"))
    pp(baseline.similarity("truly", "masterpiece"))

    # pp(baseline.most_similar_cosmul(positive=['movie'], topn=15))
    # pp(baseline.similarity("very", "good"))
    # pp(baseline.similarity("not", "very"))


def compute_score_model_bigrams(model, sentence):

    result = []

    prev = None
    for idx in xrange(len(sentence)):

        word = sentence[idx]

        if idx > 0:
            if word in model and prev in model:
                result.append((model.similarity(word, prev), (word, prev)))
            else:
                result.append((-1.0, (word, prev)))

        prev = word

    score = 0
    for res in result:
        score += res[0]
    print "Score ", score, " | Average score: ", score / (len(result) / 2)

    return result

def compute_score_model_trigrams(model, sentence):

    result = []

    if len(sentence) == 2:
        return compute_score_model_bigrams(model, sentence)

    prev1 = None
    prev2 = None
    for idx in xrange(len(sentence)):

        word = sentence[idx]

        if idx > 2:
            if word in model and prev1 in model and prev2 in model:
                prefix = model[prev1] + model[prev2]
                result.append((d.cosine(model[word], prefix), (prev2, prev1, word)))
            else:
                result.append((-1.0, (prev2, prev1, word)))

        prev2 = prev1
        prev1 = word

    score = 0
    for res in result:
        score += res[0]
    print "Score ", score, " | Average score: ", score / (len(result) / 3)

    return result


def battlefield_test():

    neg_model = w2vec_model('../data/w2v_0_corpus.model')
    pos_model = w2vec_model('../data/w2v_1_corpus.model')
    baseline  = w2vec_model('../data/w2v_corpus.model')

    mongo = MongoORM("corpus")

    for item in mongo.get_collection("test"):

        for sentence in item["sentences"]:

            pp("Sentence")
            pp(sentence)

            pp("Score for negative model: ")
            pp(compute_score_model_trigrams(neg_model, sentence))
            print

            pp("Score for positive model: ")
            pp(compute_score_model_trigrams(pos_model, sentence))
            print

            # pp("Score for baseline model: ")
            # pp(compute_score_model_trigrams(baseline, sentence))
            # print

        exit(0)


def main():

    # step 0: build corpus
    # step 1: train word vectors
    # train_word_vectors()
    # w2vec_model()
    # step 2: cluster phrases
    # cluster_phrases()
    # cluster_test()
    # store_cluster_results()

    # train_network(14)
    # retrain_network(14)
    # train_network(29)

    # Hypothesis: Improved Bayes using
    # w2vec_train_steroids()
    # w2vec_test()
    battlefield_test()


if __name__ == "__main__":
    main()
