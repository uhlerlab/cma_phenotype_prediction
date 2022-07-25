import argparse


def setup_options():
    options = argparse.ArgumentParser()
    options.add_argument('-e', action='store', dest='embedding_path',
                         default='')
    options.add_argument('-s', action='store', dest='seed',
                         default=17, type=int)
    options.add_argument('-tp', action='store', dest='target_path',
                         default='data/phenotypes.tsv'),
    options.add_argument('-ts', action='store', dest='train_set',
                         default='split_ids/train_set.h')
    options.add_argument('-vs', action='store', dest='val_set',
                         default='split_ids/val_set.h')
    options.add_argument('-ss', action='store', dest='test_set',
                         default='split_ids/test_set.h')
    options.add_argument('-k', action='store', dest='kernel',
                         default='ntk')
    options.add_argument('-n', action='store', dest='num_epochs',
                         default=30, type=int)
    return options.parse_args()
