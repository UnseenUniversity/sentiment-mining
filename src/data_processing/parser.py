

import re
import nltk
from nltk import data, corpus
from nltk.stem import WordNetLemmatizer as wnl

from stanford import StanfordNLP
from pprint import pprint

class Parser():

    def __init__(self):

        self.tokenizer = data.load('tokenizers/punkt/english.pickle')
        self.stopwords = corpus.stopwords.words("english")
        self.wnl       = wnl()

        self.stanford_parser = StanfordNLP()

        self.delims_patterns = [r".", r",", r"?", r"!", r":", r";"]
        self.delims = {".", ',', "!", ":", ";"}

    def parse_dependencies(self, dependencies):

        result = []
        components = {}

        for [type, target, source] in dependencies:

            if type is "root":
                continue

            if type[0] is 'a':
                print type, target, source

            exit(0)

    def parse_words(self, blob):

        result = {}
        for item in blob:

            word         = item[0]
            if word in self.delims:
                continue

            # box of cookies
            word_dict    = item[1]
            lemma        = word_dict["Lemma"]
            entity_tag   = word_dict["NamedEntityTag"]
            result[word] = (lemma, entity_tag)

        return result

    def parse_review_text(self, text, word_dict):

        result = []
        for token in text.split():
            if token in word_dict:
                result.append(word_dict[token][0])

        return result

    def parse_result(self, item, result):

        for sentence in result["sentences"]:

            word_dict = self.parse_words(sentence["words"])
            text = self.parse_review_text(sentence["text"], word_dict)

            if len(text) >= 2:
                item["text"].append(text)
                deps = sentence["dependencies"]
                item["deps"].append(deps)

    def parse_split(self, item, review):
        size = len(review)

        first_part, second_part = "", ""

        idx = 0
        while idx < size:
            if idx > size / 2 and review[idx] in self.delims:
                first_part, second_part = review[:idx], review[idx:]
                break
            idx += 1

        if len(first_part) > 1000:
            self.parse_split(item, first_part)
        else:
            result = self.stanford_parser.parse(first_part)
            self.parse_result(item, result)

        if len(second_part) > 1000:
            self.parse_split(item, second_part)
        else:
            result = self.stanford_parser.parse(second_part)
            self.parse_result(item, result)


    def parse_review_simple(self, review):
        review = self.remove_insanity(review)
        review = self.isolate_delims(review)
        review = review.split()
        return review

    def lemmatize(self, word, pos):

        if pos == "j" or pos == "r":
            return self.wnl.lemmatize(word, "a")

        if pos == "n":
            return self.wnl.lemmatize(word, "n")

        if pos == "v":
            return self.wnl.lemmatize(word, "v")

        return word

    def parse_carefully(self, text):

        sentences = []
        text = self.remove_insanity(text)
        text = self.remove_delims(text)
        text = text.split(".")

        for phrase in text:

            tokens = []
            sentence = nltk.word_tokenize(phrase)

            for token in sentence:
                if len(token) == 1 or token == "the":
                    continue

                pos = nltk.pos_tag([token])[0][1][0].lower()
                token = self.lemmatize(token, pos)
                tokens.append(token)

            if len(tokens) == 0:
                continue

            sentences.append(tokens)

        return sentences


    def parse_review(self, review):

        review = self.remove_insanity(review)
        review = self.isolate_delims(review)

        item = dict()
        item["text"] = []
        item["deps"] = []

        try:
            if len(review) > 1000:
                self.parse_split(item, review)
            else:
                result = self.stanford_parser.parse(review)
                self.parse_result(item, result)
        except Exception as e:
            item["text"] = self.parse_dummy(review)
            pprint("stanford parser fail")
            print review

        return item

    def remove_insanity(self, text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = self.remove_markup(text)
        text = re.sub(r"\"|\\|\*|&|-|_|/|~|`|#|@", " ", text)
        text = text.lower()
        return text

    def remove_markup(self, raw_text):
        string = re.sub(r"<(.*?)>", " ", raw_text.strip())
        return string

    def remove_delims(self, text):
        text = re.sub(r",|\(|\)", "  ", text)
        text = re.sub(r"\.|\?|!|:|;", " . ", text)
        return text

    def isolate_delims(self, text):
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\.|\?|!|:|;", " . ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ) ", text)
        return text

    def split_to_sentences(self, raw_text):
        sentences = self.tokenizer.tokenize(raw_text)
        sentences = filter(lambda x: len(x) > 0, sentences)
        return sentences

    def strip_delims(self, text):

        text = re.sub(r"\.|\\|\?|!|:|\"|\(|\)|-|&| / | - |,|_|;|\*", " ", text)
        # text = re.sub(r"'s", " 's", text)
        return text

    def token_interesting_digit(self, token):
        return token.isdigit() and int(token) > 10

    def split_tokens(self, text):

        tokens = text.split()
        tokens = [tok.lower() for tok in tokens if not (len(tok) == 1)]
        return tokens

    def concat_names(self, tokens):

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

        return result

    def split_sentence(self, phrase):

        phrase = self.strip_delims(phrase)
        tokens = self.split_tokens(phrase)

    def clean_phrase(self, sentence):

        sentence = self.strip_delims(sentence)
        tokens = self.split_tokens(sentence)
        result = self.concat_names(tokens)

        return result

    def parse_dummy(self, text):

        text = self.remove_insanity(text)
        sentences = self.tokenizer.tokenize(text)
        sentences = filter(lambda x: len(x) > 0, sentences)
        sentences = map(self.clean_phrase, sentences)

        return sentences

