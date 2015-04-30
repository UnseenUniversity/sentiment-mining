__author__ = 'alexei'

import pprint as pp
from data_processing.util import *
from data_processing.mongo import MongoORM

import nltk
from nltk.stem import WordNetLemmatizer as wnl
wnl = wnl()
ignore_set = {"\\"}


import os
from nltk.parse import stanford

stanford_path = '/home/alexei/stanford-parser-full-2015-04-20'
os.environ['STANFORD_PARSER'] = stanford_path
os.environ['STANFORD_MODELS'] = stanford_path


def test_deps():
    parser = stanford.StanfordParser(model_path=stanford_path + "/englishPCFG.ser.gz")
    sentences = parser.raw_parse_sents(("My true name isn't Melroy and Dan.", "What is your name?"))
    for line in sentences:
        for sentence in line:
            sentence.draw()



def lemmatize_word(word, word_set):

    if word not in word_set:
        pos = nltk.pos_tag([word])[0][1][0].lower()
        if pos in {'n', 'a', 'v'}:
            return wnl.lemmatize(word, pos)
    return word

def process_word(word, word_set, feature_idx):

    word = lemmatize_word(word, word_set)
    if word not in feature_idx:
        feature_idx[word] = FeatureNode(word)
    return word, feature_idx[word]


class FeatureNode:

    def __init__(self, word):
        self.word = word
        self.parent = self
        self.rank   = 1
        self.modifiers = set()
        self.negation = False
        self.deps = set()
        self.interesting = False

    def find_parent(self):
        parent = self.parent
        while parent != parent.parent:
            parent = parent.parent

        it_parent = self.parent
        while it_parent.parent != parent:
            next_node = it_parent.parent
            it_parent.parent = parent
            it_parent = next_node

        return parent


    def join_set(self, other):

        parent_1 = self.find_parent()
        parent_2 = other.find_parent()

        if parent_1 != parent_2:
            if self.rank > other.rank:
                self.rank += other.rank
                parent_2.parent = parent_1
            else:
                other.rank += self.rank
                parent_1.parent = parent_2

    def same_set(self, other):
        return self.find_parent() == other.find_parent()

    def append_modifier(self, modifier):
        self.modifiers.add(modifier)
        self.interesting = True

    def negate(self):
        self.negation = True
        self.interesting = True

    def add_dependency(self, dep):
        self.deps.add(dep)
        self.interesting = True

def disjoint_sets_test():
    feats = []
    for f in xrange(5):
        feats.append(FeatureNode(f))

    feats[1].join_set(feats[2])
    feats[3].join_set(feats[4])

    if feats[1].same_set(feats[3]):
        print "DA"
    else:
        print "NU"

    if feats[1].same_set(feats[2]):
        print "DA"
    else:
        print "NU"

    feats[1].join_set(feats[3])

    if feats[1].same_set(feats[4]):
        print "DA"
    else:
        print "NU"


from pprint import pprint as pp
def process_review(text, deps):

    idx = 0
    num_sentences = len(text)

    review_features = []

    while idx < num_sentences:

        sentence = text[idx]
        dep      = deps[idx]

        lexem_dict = {}
        lexem_set = set(sentence)
        feature_idx = {}

        pp(dep)

        for d in dep:
            if d[0] == "root" or d[1] in ignore_set or d[2] in ignore_set:
                continue

            d[1], fnode1 = process_word(d[1], lexem_set, feature_idx)
            d[2], fnode2 = process_word(d[2], lexem_set, feature_idx)

            if d[0] == "conj_and":
                fnode1.join_set(fnode2)

        for d in dep:
            if d[0] == "root" or d[1] in ignore_set or d[2] in ignore_set:
                continue

            fnode1 = feature_idx[d[1]]
            if d[0] == "neg":
                fnode1.parent.negate()


        for d in dep:
            if d[0] == "root" or d[1] in ignore_set or d[2] in ignore_set:
                continue

            fnode1 = feature_idx[d[1]]
            fnode2 = feature_idx[d[2]]

            if d[0] in {"advmod", "advcl", "amod"}:
                fnode1.parent.append_modifier(fnode2)
            elif d[0] == "acomp" and fnode2.parent.negation:
                fnode1.parent.negate()
            elif d[0] == "neg":
                continue
            elif d[0] == "prep" or d[0] == "punct" or d[0] == "det":
                continue
            else:
                fnode1.parent.add_dependency(fnode2)


        for word in feature_idx:
            feature = feature_idx[word]

            if idx == 3:
                print "word ", word, " | ", feature.parent.word

            if not feature.parent.interesting:
                continue

            if idx == 3:
                print "feature", feature.word

            feature_set = []

            if len(feature.parent.modifiers) == 0:
                feature_set.append({word})
            else:
                for mod in feature.parent.modifiers:
                    feature_set.append({mod.word, word})

            if idx == 3:
                print "sets with mods", feature_set

            if feature.parent.negation:
                for feat in feature_set:
                    feat.add("neg")

            if idx == 3:
                print "sets with negs", feature_set

            new_features = []
            for dep in feature.parent.deps:
                for feat in feature_set:
                    new_features.append(feat | {dep.word})

            feature_set += new_features
            review_features += feature_set

            if idx == 3:
                pp(feature_set)

        print
        if idx == 3:
            pp(review_features)
            exit(0)

        # review_features += feature_set



        idx += 1

    pp(review_features)
    exit(0)


    return None







def build_imdb_dataset():

    mongo = MongoORM("imdb")
    labeled = mongo.get_collection("labeled")

    t = Timer()

    result = []

    for review in labeled:
        label = review["sentiment"]
        deps  = review["deps"]
        text  = review["text"]

        features = process_review(text, deps)
        # result += features

    t.measure("imdb dataset build in: ")

    return result

def main():

    train = build_imdb_dataset()

    # test_deps()





if __name__ == "__main__":
    main()