__author__ = 'alexei'

from multiprocessing import Pool
from datetime import datetime, timedelta
import re

class Proc():

    def __init__(self, num_cores=4):
        self.num_cores = num_cores
        self.pool = Pool(processes=num_cores)

    def compute(self, task, params):
        result = self.pool.map(task, params)
        return result


class Timer():

    def __init__(self):
        self.trigger()

    def trigger(self):
        self.start = datetime.now()

    def measure(self, msg="Time elapsed"):
        print msg + "%s" % (datetime.now() - self.start)


def extract_data(source, delim="\t"):

    path = "../data"

    if source == "labeled":
        path += "/labeledTrainData.tsv"
    elif source == "test":
        path += "/testData.tsv"
    elif source == "unlabeled":
        path += "/unlabeledTrainData.tsv"
    elif source == "rotten":
        path += "/rotten_train.tsv"
    else:
        path += "/" + source

    data = {}
    tags = []
    for e, line in enumerate(open(path, "rb")):
        content = re.split(delim, line.strip())

        if e == 0:
            for tag in content:
                data[tag] = []
                tags.append(tag)
        else:
            for idx in xrange(len(tags)):
                data[tags[idx]].append(content[idx])

    return data

from nltk import data, corpus
from nltk.stem import WordNetLemmatizer as wnl


class Parser():

    def __init__(self):

        self.tokenizer = data.load('tokenizers/punkt/english.pickle')
        self.stopwords = corpus.stopwords.words("english")
        self.wnl       = wnl()

    def clean_phrase(self, phrase):

        #remove delimitators
        phrase = re.sub(r"\.|\\|\?|!|:|\"|\(|\)|-|&| / | - |,|_|;|\*", " ", phrase)
        phrase = re.sub(r"'s", " 's", phrase)

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


def build_dataset(parser, datatype, inject=False, merge=False):

    t = Timer()

    data = extract_data(datatype)

    data_set = []

    if datatype == "rotten":

        num_phrases = len(data["PhraseId"])

        sent_dict = set()

        for idx in xrange(num_phrases):
            sent_id = data["SentenceId"][idx]

            if sent_id not in sent_dict:
                sent_dict.add(sent_id)
                phrase = parser.clean_phrase(data["Phrase"][idx])
                data_set += [phrase]

    else:
        num_reviews = len(data["review"])

        for idx in xrange(num_reviews):
            sentences = parser.extract_terms(data["review"][idx])

            if merge:
                data_set += sentences
            else:
                data_set +=

            return data_set

        t.measure(datatype + " dataset built in: ")

    return data_set


def load_words_prior():

    negative = [line.strip() for line in open('../data/negative-words.txt')]
    positive = [line.strip() for line in open('../data/positive-words.txt')]

    prior_dict = {}

    for word in positive:
        prior_dict[word] = 1

    for word in negative:
        prior_dict[word] = 0

    return prior_dict