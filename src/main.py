__author__ = 'alexei'

import data_processing.data_proc as dp
import data_processing.wordvec_train as wv_train

def prepare_data():

    # dp.dump_rotten_dataset()
    # dp.dump_rotten_unigrams()
    dp.dump_unigrams_from_clusters()
    # dp.dump_imdb_dataset()
    # dp.dump_imdb_dataset_unlabeled()
    # wv_train.build_corpus()











def main():
    prepare_data()


if __name__ == "__main__":
    main()