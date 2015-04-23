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
            for idx in xrange(len(tags)):
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
            for idx in xrange(len(tags)):
                item[tags[idx]] = content[idx]
            fun(item)



def parse_item_imdb(parser, mongo, collection, item):

    _id = re.sub("\"", "", item["id"])
    if mongo.get_item_by_id(collection, _id):
        return

    result = parser.parse_review(item["review"])
    result["sentiment"] = int(item["sentiment"])
    result["_id"] = _id

    pprint(_id)

    mongo.upsert_item(collection, result)



def dump_corpus(mongo, path, collection):

    parser = p.Parser()
    map_fun = lambda item: parse_item_imdb(parser, mongo, collection, item)
    parse_csv(path, map_fun)



def dump_imdb_dataset():

    mongo = MongoORM("imdb")
    dump_corpus(mongo, "../data/labeledTrainData.tsv", "labeled")
    dump_corpus(mongo, "../data/unlabeledTrainData.tsv", "unlabeled")
    dump_corpus(mongo, "../data/testData.tsv", "test")

