__author__ = 'alexei'

import gensim
from mongo import MongoORM
from util import Timer

def rotten_corpus(result):

    t = Timer()
    mongo = MongoORM("rotten")
    phrases = mongo.get_collection("phrases")

    prev_sent_id = -1

    idx = 0
    for phrase in phrases:
        sent_id = phrase["sent_id"]

        if sent_id == prev_sent_id:
            continue

        prev_sent_id = phrase["sent_id"]
        sentences = phrase["text"]
        result += sentences
    t.measure("rotten corpus built in: ")


    t = Timer()
    phrases = mongo.get_collection("test_phrases")
    prev_sent_id = -1

    idx = 0
    for phrase in phrases:
        sent_id = phrase["sent_id"]

        if sent_id == prev_sent_id:
            continue

        prev_sent_id = phrase["sent_id"]
        sentences = phrase["text"]
        result += sentences

    t.measure("rotten test corpus built in: ")
    return result

def imdb_corpus(result):

    t = Timer()
    mongo = MongoORM("imdb")
    labeled = mongo.get_collection("labeled")
    for item in labeled:
        result += item["text"]

    unlabeled = mongo.get_collection("unlabeled")
    for item in unlabeled:
        result += item["text"]

    t.measure("imdb corpus built in: ")

def build_corpus():

    result = []
    rotten_corpus(result)
    imdb_corpus(result)

    print "Number of sentences ", len(result)
    return result


def w2vec_train(sentences):

    print "Train w2vec model: "

    t = Timer()
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=2, workers=4)
    model.save('./w2v_for_rotten.model')
    t.measure("word2vec model built in: ")

def w2vec_model():

    model = gensim.models.Word2Vec.load('./w2v_for_rotten.model')
    model.init_sims(replace=True)

    print model.most_similar_cosmul(positive=["great"], topn=20)
    # print model.most_similar_cosmul(positive=["terrible"], topn=20)
    # print model.most_similar_cosmul(positive=["john"], topn=10)
    # print model.most_similar_cosmul(positive=["good"], negative=["actor"], topn=20)

def main():
    # result = build_corpus()
    # w2vec_train(result)
    w2vec_model()

if __name__ == "__main__":
    main()