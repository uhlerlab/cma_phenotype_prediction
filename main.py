import numpy as np
import dataset
import torch
import eigenpro
from scipy.stats import pearsonr
import kernel
from numpy.linalg import norm
import options_parser as op
import hickle
import csv
import random


def ntk_kernel(pair1, pair2):

    out = pair1 @ pair2.transpose(1, 0) + 1
    N1 = torch.sum(torch.pow(pair1, 2), dim=-1).view(-1, 1) + 1
    N2 = torch.sum(torch.pow(pair2, 2), dim=-1).view(-1, 1) + 1

    XX = torch.sqrt(N1 @ N2.transpose(1, 0))
    out = out / XX

    out = torch.clamp(out, -1, 1)
    first = 1/np.pi * (out * (np.pi - torch.acos(out)) \
                       + torch.sqrt(1. - torch.pow(out, 2))) * XX
    sec = 1/np.pi * out * (np.pi - torch.acos(out)) * XX
    out = first + sec

    # 4 is just a constant factor to reduce learning rate with eigenpro
    return out / 4


def linear_kernel(pair1, pair2):
    out = pair1 @ pair2.transpose(1, 0)
    return out


def get_kernel_fn(flag):
    if flag == 'linear':
        return linear_kernel
    elif flag == 'laplace':
        return lambda x,y: kernel.laplacian(x, y, bandwidth=5)
    else:
        return ntk_kernel


def main(options):
    seed = options.seed
    embedding_path = options.embedding_path
    kernel_flag = options.kernel
    np.random.seed(seed)

    target_path = options.target_path
    train_ids = hickle.load(options.train_set)
    val_ids = hickle.load(options.val_set)
    test_ids = hickle.load(options.test_set)
    splits = dataset.make_dataset(embedding_path, target_path,
                                  [train_ids, val_ids, test_ids],
                                  seed=seed)
    train_X, train_y, val_X, val_y, test_X, test_y, phenotypes, all_ids = splits
    train_ids, val_ids, test_ids = all_ids
    print(len(train_ids), len(val_ids), len(test_ids))

    num_examples = -1
    train_X = train_X[:num_examples]
    train_y = train_y[:num_examples]

    norm_train = norm(train_X, 2, axis=-1).reshape(-1, 1)
    train_X /= norm_train
    norm_val = norm(val_X, 2, axis=-1).reshape(-1, 1)
    val_X /= norm_val
    norm_test = norm(test_X, 2, axis=-1).reshape(-1, 1)
    test_X /= norm_test

    print("DATASET: ")
    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)
    print("KERNEL: ", kernel_flag)

    kernel_fn = get_kernel_fn(kernel_flag)

    num_classes = 1
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = eigenpro.FKR_EigenPro(kernel_fn, train_X, train_y.shape[-1], device=device)
    MAX_EPOCHS = options.num_epochs

    epochs = list(range(MAX_EPOCHS))
    metrics = model.fit(train_X, train_y, val_X, val_y, epochs=epochs, mem_gb=12)

    test = torch.from_numpy(test_X)
    best_r2 = -2
    for idx, key in enumerate(metrics):
        if key != 0:
            new_r2 = sum(metrics[key][-1].values())/len(metrics[key][-1])
            if new_r2 > best_r2:
                best_r2 = new_r2
                best_weight = metrics[key][0]

    print("BEST R2: ", best_r2)
    model.weight = best_weight

    val_r = model.evaluate(val_X, val_y, 256)['r2']
    test_r = model.evaluate(test_X, test_y, 256)['r2']
    print(val_r, test_r)

    with open('csv_outputs/output.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for key in val_r:
            writer.writerow([phenotypes[key],
                            val_r[key],
                             test_r[key]])
        writer.writerow(['Average', sum(val_r.values()) / len(val_r),
                         sum(test_r.values()) / len(test_r)])
    print("AVERAGE VAL R:", sum(val_r.values()) / len(val_r))
    print("AVERAGE TEST R:", sum(test_r.values()) / len(test_r))


options = op.setup_options()
if __name__ == "__main__":
    main(options)
