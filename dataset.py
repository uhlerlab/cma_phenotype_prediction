import numpy as np
import csv
import hickle
import pandas as pd
from sklearn.utils import shuffle


def make_dataset(embedding_path, target_path, splits, seed=17):
    embed_df = load_embeddings(embedding_path, '\t')
    target_df = load_targets(target_path, '\t')

    train, val, test = generate_data(embed_df,
                                     target_df,
                                     splits)

    # Example for MRI derived phenotypes
    phenotypes = ['LVM', 'LVEDV', 'LVEF',
                  'LVESV', 'LVSV', 'RVEF', 'RVESV',
                  'RVSV', 'RVEDV']


    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test

    train_y = train_y[phenotypes]
    val_y = val_y[phenotypes]
    test_y = test_y[phenotypes]

    train_X, train_y, train_ids = trim_na(train_X, train_y)
    val_X, val_y, val_ids = trim_na(val_X, val_y)
    test_X, test_y, test_ids = trim_na(test_X, test_y)

    train_X, train_y = shuffle(train_X, train_y, random_state=seed)

    # Normalizing Labels
    mean, std = np.mean(train_y, axis=0), np.std(train_y, axis=0)
    train_y = (train_y - mean) / std
    val_y = (val_y - mean) / std
    test_y = (test_y - mean) / std
    print("Mean: ", mean)
    print("Std: ", std)

    print("Train Set: ", train_X.shape, train_y.shape)
    print("Val Set: " , val_X.shape, val_y.shape)
    print("Test Set: ", test_X.shape, test_y.shape)

    all_ids = (train_ids, val_ids, test_ids)
    return train_X, train_y, val_X, val_y, test_X, test_y, phenotypes, all_ids

def trim_na(X, y):
    na_ids = y[y.isna().any(axis=1)].index.values
    y = y.drop(na_ids)
    X = X.drop(na_ids)
    ids = list(X.index)
    return np.array(X).astype('float32'), np.array(y).astype('float32'), ids

def map_data(ids, embed_df, target_df):

    ids = [idx for idx in ids]

    X = embed_df.loc[ids].sort_values(by=['sample_id'])
    y = target_df.loc[ids].sort_values(by=['sample_id'])
    return X, y


def generate_data(embed_df, target_df, splits):
    train_ids, val_ids, test_ids = splits
    train = map_data(train_ids, embed_df, target_df)
    val = map_data(val_ids, embed_df, target_df)
    test = map_data(test_ids, embed_df, target_df)

    return train, val, test


def load_embeddings(embedding_path, delimiter):
    df = pd.read_csv(embedding_path, sep=delimiter)
    df = df.set_index('sample_id')

    return df


def load_targets(target_path, delimiter):
    df = pd.read_csv(target_path, sep=delimiter)
    df = df.set_index('sample_id')
    return df
