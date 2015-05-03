__author__ = 'alexei'

import re
from mongo import MongoORM
import parser as p
from pprint import pprint

def read_csv(path, delim='\t'):

    data = {}
    tags = []

    for e, line in enumerate(open(path, "rb")):

        content = line.strip().split(delim)

        if e == 0:
            for tag in content:
                data[tag] = []
                tags.append(tag)
        else:
            length = min(len(tags), len(content))
            for idx in xrange(length):
                data[tags[idx]].append(content[idx])

    return data

def parse_csv(path, fun, delim='\t'):

    tags = []
    for e, line in enumerate(open(path, "rb")):

        content = line.strip().split(delim)

        if e == 0:
            tags = [tag for tag in content]
        else:
            item = {}

            #bad file
            if len(content) < len(tags):
                item[tags[len(tags) - 1]] = ""

            for idx in xrange(len(content)):
                item[tags[idx]] = content[idx]

            fun(item)

import nltk
from nltk.stem import WordNetLemmatizer as wnl
wnl = wnl()

def lemmatize_word(word):
    pos = nltk.pos_tag([word])[0][1][0].lower()
    if pos in {'n', 'a', 'v'}:
        return wnl.lemmatize(word, pos)
    return word

def dump_unigrams_from_clusters():

    delim = "\t"
    path  = "../data/full_clusters.csv"

    mongo = MongoORM("words")

    for e, line in enumerate(open(path, "rb")):
        content = line.strip().split(delim)
        label   = int(content[0])
        if label != 0:
            for word in content[1:]:
                if label == 1:
                    mongo.update_item("positive", {"_id": word}, {"$set": {"label": 4}})
                    # mongo.upsert_item("positive", {"_id": word, "label": 4})
                elif label == -1:
                    # mongo.upsert_item("negative", {"_id": word, "label": 0})
                    mongo.update_item("negative", {"_id": word}, {"$set": {"label": 0}})

def parse_item_imdb_repair(parser, mongo, collection, item):

    _id = re.sub("\"", "", item["id"])
    result = parser.parse_review_simple(item["review"])

    print _id, result
    exit(0)
    mongo.update_item(collection, {"_id": _id}, {"orig_text": result})


def parse_item_imdb(parser, mongo, collection, item):

    _id = re.sub("\"", "", item["id"])
    if mongo.get_item_by_id(collection, _id):
        return

    result = parser.parse_review(item["review"])
    result["sentiment"] = int(item["sentiment"])
    result["_id"] = _id

    pprint(_id)

    mongo.upsert_item(collection, result)




def classify_words_rotten(mongo, item):

    text = item["Phrase"]
    words = text.split()

    sentiment = int(item["Sentiment"])
    if len(words) == 1 and sentiment != 2:

        word = lemmatize_word(words[0].lower())
        if sentiment < 2:
            mongo.upsert_item("negative", {"_id": word, "label": sentiment})
        else:
            mongo.upsert_item("positive", {"_id": word, "label": sentiment})


def dump_unigram_classifier(mongo, path):
    map_fun = lambda item: classify_words_rotten(mongo, item)
    parse_csv(path, map_fun)

def dump_rotten_unigrams():
    mongo = MongoORM("words")
    dump_unigram_classifier(mongo, "../data/rotten_train.tsv")

count = 0

def parse_item_rotten(parser, mongo, collection, item):

    global count
    count += 1
    if count <= 1390:
        return

    _id = int(item["PhraseId"])

    # if mongo.get_item_by_id(collection, _id):
    #     return

    if len(item["Phrase"]) > 0:
        result = parser.parse_review(item["Phrase"])
    else:
        result = dict()
        result["text"] = []
        result["deps"] = []

    # if "Sentiment" in item:
    #     result["sentiment"] = int(item["Sentiment"])

    result["sent_id"] = int(item["SentenceId"])
    result["_id"] = _id

    if (_id + 1) % 100 == 0:
        pprint(_id)

    mongo.insert_item(collection, result)


def dump_corpus_imdb(mongo, path, collection):

    parser = p.Parser()
    map_fun = lambda item: parse_item_imdb_repair(parser, mongo, collection, item)
    parse_csv(path, map_fun)


def dump_imdb_dataset():

    mongo = MongoORM("imdb")
    dump_corpus_imdb(mongo, "../data/labeledTrainData.tsv", "labeled")


def dump_corpus_rotten(mongo, path, collection):

    parser = p.Parser()
    map_fun = lambda item: parse_item_rotten(parser, mongo, collection, item)
    parse_csv(path, map_fun)


def dump_rotten_dataset():

    mongo = MongoORM("rotten")
    # dump_corpus_rotten(mongo, "../data/rotten_train.tsv", "phrases")
    dump_corpus_rotten(mongo, "../data/rotten_test.tsv", "test_phrases")




def parse_item_unlabeled(parser, mongo, collection, item):

    _id = re.sub("\"", "", item["id"])
    if mongo.get_item_by_id(collection, _id):
        return

    result = parser.parse_review(item["review"])
    del result["deps"]
    result["_id"] = _id

    pprint(_id)
    mongo.upsert_item(collection, result)


def dump_corpus_imdb_unlabeled(mongo, path, collection):

    print "Dump corpus ", collection

    parser = p.Parser()
    map_fun = lambda item: parse_item_unlabeled(parser, mongo, collection, item)
    parse_csv(path, map_fun)


def dump_imdb_dataset_unlabeled():

    mongo = MongoORM("imdb")
    dump_corpus_imdb_unlabeled(mongo, "../data/unlabeledTrainData.tsv", "unlabeled")
    dump_corpus_imdb_unlabeled(mongo, "../data/testData.tsv", "test")


####################################################

import util

def parse_imdb_review(parser, mongo, collection, item):

    _id = re.sub("\"", "", item["id"])
    sentences = parser.parse_carefully(item["review"])

    result = {"_id": _id,
              "sentences": sentences}

    if "sentiment" in item:
        result["label"] = item["sentiment"]

    mongo.upsert_item(collection, result)


def dump_imdb2(parser, mongo, path, collection):

    t = util.Timer()
    map_fun = lambda item: parse_imdb_review(parser, mongo, collection, item)
    parse_csv(path, map_fun)
    t.measure(collection + " corpus built in : ")


def build_imdb_corpus():

    mongo = MongoORM("corpus")
    parser = p.Parser()
    dump_imdb2(parser, mongo, "../data/labeledTrainData.tsv", "labeled")
    dump_imdb2(parser, mongo, "../data/unlabeledTrainData.tsv", "unlabeled")
    dump_imdb2(parser, mongo, "../data/testData.tsv", "test")


prev_sent_id = -1


def parse_rotten_sentence(parser, mongo, collection, item):

    global prev_sent_id
    sent_id = int(item["SentenceId"])

    if sent_id == prev_sent_id:
        return

    prev_sent_id = sent_id

    if len(item["Phrase"]) == 0:
        return

    _id       = int(item["PhraseId"])
    sentences = parser.parse_carefully(item["Phrase"])

    result = {"_id": _id,
              "sentences": sentences}

    if "Sentiment" in item:
        label = int(item["Sentiment"])
        result["label"] = label

    mongo.upsert_item(collection, result)


def dump_rotten2(parser, mongo, path, collection):

    global prev_sent_id
    prev_sent_id = -1

    t = util.Timer()
    map_fun = lambda item: parse_rotten_sentence(parser, mongo, collection, item)
    parse_csv(path, map_fun)
    t.measure(collection + " corpus built in: ")


def build_rotten_corpus():

    mongo = MongoORM("corpus")
    parser = p.Parser()
    dump_rotten2(parser, mongo, "../data/rotten_train.tsv", "rotten_train")
    dump_rotten2(parser, mongo, "../data/rotten_test.tsv", "rotten_test")
