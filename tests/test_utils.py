import warnings

import numpy as np
import pytest
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
