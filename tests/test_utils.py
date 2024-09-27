import warnings

import numpy as np
import pytest
import scanpy as sc
import torch
from utils import generate_dummy_adata

import nichepca as npc


def test_to_numpy():
    adata = generate_dummy_adata()
    X_sparse = adata.X

    X_np = X_sparse.toarray()

    X_torch = torch.tensor(X_np)

    assert np.all(npc.utils.to_numpy(X_sparse) == X_np)
    assert np.all(npc.utils.to_numpy(X_np) == X_np)
    assert np.all(npc.utils.to_numpy(X_torch) == X_np)


def test_to_torch():
    adata = generate_dummy_adata()
    X_sparse = adata.X

    X_np = X_sparse.toarray()

    X_torch = torch.tensor(X_np)

    assert (npc.utils.to_torch(X_sparse) == X_torch).all()
    assert (npc.utils.to_torch(X_np) == X_torch).all()
    assert (npc.utils.to_torch(X_torch) == X_torch).all()


def test_check_for_raw_counts():
    adata = generate_dummy_adata()

    # Check that no warnings are raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        npc.utils.check_for_raw_counts(adata)

        # Ensure no warnings were raised
        assert len(w) == 0

    adata.X = adata.X / 100
    # Check for the specific warning
    with pytest.warns(UserWarning):
        npc.utils.check_for_raw_counts(adata)


def test_normalize_per_sample():
    sample_key = "sample"

    target_sum = 1e4

    adata_1 = generate_dummy_adata()
    npc.utils.normalize_per_sample(
        adata_1, target_sum=target_sum, sample_key=sample_key
    )

    adata_2 = generate_dummy_adata()
    sc.pp.normalize_total(adata_2, target_sum=target_sum)

    assert np.all(adata_1.X.toarray() == adata_2.X.toarray())

    # second test without fixed target sum
    target_sum = None

    adata_1 = generate_dummy_adata()
    npc.utils.normalize_per_sample(
        adata_1, target_sum=target_sum, sample_key=sample_key
    )

    adata_2 = generate_dummy_adata()
    adata_2.X = adata_2.X.astype(np.float32).toarray()

    for sample in adata_2.obs[sample_key].unique():
        mask = adata_2.obs[sample_key] == sample
        sub_ad = adata_2[mask].copy()
        sc.pp.normalize_total(sub_ad)
        adata_2.X[mask.values] = sub_ad.X

    assert np.all(adata_1.X.astype(np.float32).toarray() == adata_2.X)
