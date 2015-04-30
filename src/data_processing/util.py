__author__ = 'alexei'

from multiprocessing import Pool
from datetime import datetime, timedelta

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
