from src.modules.data.data_splitter import FixedSplitter, RandomSplitter, StratifiedSplitter
from collections import Counter
from torch_geometric.datasets import FakeDataset
import torch
import numpy as np


def test_fixed_splitter():
    """Unit tests for fixed splitting"""
    dset = FakeDataset(num_graphs=5)
    num_cases = len(dset)

    # fixed set of indices
    train_idx = [0, 1]
    val_idx = [2, 3, 4]
    test_idx = list(range(5, num_cases))

    splitter = FixedSplitter(split=[train_idx, val_idx, test_idx])
    set_train, set_val, set_test = splitter.run(dset)

    assert len(train_idx) == len(set_train)
    assert len(val_idx) == len(set_val)
    assert len(test_idx) == len(set_test)

    # check equality of the data
    for i, s in zip(train_idx, set_train):
        torch.testing.assert_close(s.x, dset[i].x)
        torch.testing.assert_close(s.y, dset[i].y)
        torch.testing.assert_close(s.edge_index, dset[i].edge_index)

    for i, s in zip(val_idx, set_val):
        torch.testing.assert_close(s.x, dset[i].x)
        torch.testing.assert_close(s.y, dset[i].y)
        torch.testing.assert_close(s.edge_index, dset[i].edge_index)

    for i, s in zip(test_idx, set_test):
        torch.testing.assert_close(s.x, dset[i].x)
        torch.testing.assert_close(s.y, dset[i].y)
        torch.testing.assert_close(s.edge_index, dset[i].edge_index)


def test_random_splitter():
    """Unit tests for random splitting
    """
    dset = FakeDataset(num_graphs=100, task="graph")

    # create a random splitter
    splitter = RandomSplitter(split=[0.2, 0.3, 0.5])
    set_train, set_val, set_test = splitter.run(dset)

    # check if all cases have been assigned
    assert len(dset) == (len(set_test) + len(set_val) + len(set_train))


def test_stratified_splitter():
    """Unit tests for stratified splitting"""
    dset = FakeDataset(num_graphs=1000, task="graph", num_classes=5)

    counter_all = Counter([ds.y.item() for ds in dset])
    bal_all = {key: val / len(dset) for key, val in counter_all.items()}

    # create stratified splitter
    splitter = StratifiedSplitter(split=[0.6, 0.1, 0.3])
    set_train, set_val, set_test = splitter.run(dset)

    # check the balance of each set
    counter_train = Counter([ds.y.item() for ds in set_train])
    bal_train = {key: val / len(set_train) for key, val in counter_train.items()}
    assert all([np.isclose(bal_train[k], bal_all[k], rtol=0.01, atol=0.01) for k in bal_train.keys()])

    counter_val = Counter([ds.y.item() for ds in set_val])
    bal_val = {key: val / len(set_val) for key, val in counter_val.items()}
    assert all([np.isclose(bal_val[k], bal_all[k], rtol=0.01, atol=0.01) for k in bal_val.keys()])

    counter_test = Counter([ds.y.item() for ds in set_test])
    bal_test = {key: val / len(set_test) for key, val in counter_test.items()}
    assert all([np.isclose(bal_test[k], bal_all[k], rtol=0.01, atol=0.01) for k in bal_test.keys()])

