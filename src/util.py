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
