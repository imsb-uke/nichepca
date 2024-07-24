from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import scipy.sparse as sp
import torch

if TYPE_CHECKING:
    from anndata import AnnData


def to_numpy(X: np.ndarray | sp.spmatrix | torch.Tensor):
    """
    Convert input to numpy array.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix, torch.Tensor]
        Input data.

    Returns
    -------
    np.ndarray
        Numpy array.
    """
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    elif sp.issparse(X):
        return X.toarray()
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise ValueError(f"Unsupported type: {type(X)}")


def to_torch(X: np.ndarray | sp.spmatrix | torch.Tensor):
    """
    Convert input to torch tensor.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix, torch.Tensor]
        Input data.

    Returns
    -------
    torch.Tensor
        Torch tensor.
    """
    if isinstance(X, torch.Tensor):
        return X
    elif sp.issparse(X):
        return torch.tensor(X.toarray())
    elif isinstance(X, np.ndarray):
        return torch.tensor(X)
    else:
        raise ValueError(f"Unsupported type: {type(X)}")


def check_for_raw_counts(adata: AnnData):
    """
    Check whether `adata` contains raw counts.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.

    Returns
    -------
    None
    """
    max_val = adata.X.max()
    sum_val = adata.X.sum()

    if not max_val.is_integer() or not sum_val.is_integer():
        warn(
            f"adata.X might not contain raw counts!\nadata.X.max() = {max_val}, adata.X.sum() = {sum_val}",
            UserWarning,
            stacklevel=1,
        )
