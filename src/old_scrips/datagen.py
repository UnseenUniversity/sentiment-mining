from __future__ import division
__author__ = 'alexei'

from os import listdir
from os.path import isfile, join
import util as p


def reference_labels():
    ref = {}
    ref_data = p.extract_data("reference.csv", delim=",")
    num_reviews = len(ref_data["id"])
    for idx in xrange(num_reviews):
        ref[ref_data["id"][idx]] = ref_data["sentiment"][idx]

    return ref


def compute_score(dataset):

    ref = {}
    ref_data = p.extract_data("reference.csv")
    num_reviews = len(ref_data["id"])
    for idx in xrange(num_reviews):
        ref[ref_data["id"][idx]] = ref_data["sentiment"][idx]

    data = p.extract_data(dataset)

    label_match = [0.0, 0.0]
    label_total = [0.0, 0.0]

    matched = 0.0
    total   = 0.0
    for idx in xrange(num_reviews):

        label = int(ref[data["id"][idx]])
        label_total[label] += 1

        if label == int(data["sentiment"][idx]):
            matched += 1
            label_match[label] += 1

        total += 1

    print "Matched, Total, Perc"
    print matched, total, matched / total

    print "Negative label perc"
    print label_match[0] / label_total[0]

    print "Positive label perc"
    print label_match[1] / label_total[1]


def build_reference_dataset():

    path = "/home/alexei/Desktop/repos/sentiment-mining/aclImdb/test"

    neg_path = path + "/neg"
    neg_files = [f.replace(".txt", "") for f in listdir(neg_path) if isfile(join(neg_path, f))]

    pos_path  = path + "/pos"
    pos_files = [f.replace(".txt", "") for f in listdir(pos_path) if isfile(join(pos_path, f))]

    with open("../data/reference.csv","wb") as outfile:
        outfile.write('"id","sentiment"'+"\n")

        for filename in neg_files:
            outfile.write("%s,%s\n" % (filename, 0))

        for filename in pos_files:
            outfile.write("%s,%s\n" % (filename, 1))


def main():

    # build_reference_dataset()
    compute_score("perceptron-dummy.csv")



if __name__ == "__main__":
    main()