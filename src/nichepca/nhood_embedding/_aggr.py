from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
import torch
from torch_geometric.nn import MessagePassing

from nichepca.utils import to_numpy

if TYPE_CHECKING:
    from anndata import AnnData


class GraphAggregation(MessagePassing):
    """
    Aggregation layer for graph data using PyG.

    Parameters
    ----------
    aggr : str
        Aggregation method. Default is "mean".
    """

    def __init__(self, aggr: str = "mean"):
        super().__init__(aggr=aggr)

    def forward(self, x: torch.tensor, edge_index: torch.tensor, **kwargs):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


def aggregate(
    adata: AnnData,
    obsm_key: str | None = None,
    suffix: str = "_agg",
    n_layers: int = 1,
    out_key: str | None = None,
    backend: str = "pyg",
    aggr="mean",
):
    """
    Aggregate data in an AnnData object based on a previously constructed graph.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the data to be aggregated.
    obsm_key : str, optional
        The key in the `obsm` attribute of `adata` that points to the matrix to be aggregated.
        If None, the default matrix in `adata` will be used.
    suffix : str, default "_agg"
        Suffix to be added to the keys of the aggregated results.
    n_layers : int, default 1
        Number of layers to aggregate. This could represent different levels of aggregation or
        hierarchical aggregation.
    out_key : str, optional
        Key to store the aggregated results in the AnnData object. If None, a default key
        with the provided suffix will be used.
    backend : str, default "pyg"
        Backend to use for aggregation. Options might include "pyg" (PyTorch Geometric) or
        "sparse" (scipy.sparse).
    **kwargs : dict
        Additional keyword arguments to be passed to the aggregation backend or method.

    Returns
    -------
    None
    """
    if obsm_key is None:
        X = adata.X if backend == "sparse" else to_numpy(adata.X)
    else:
        X = adata.obsm[obsm_key]

    if backend == "pyg":
        aggr_fn = GraphAggregation(aggr)

        X = torch.tensor(X).float()
        edge_index = torch.tensor(adata.uns["graph"]["edge_index"])
        edge_weight = torch.tensor(adata.uns["graph"]["edge_weight"])

        for _ in range(n_layers):
            X = aggr_fn(X, edge_index, edge_weight=edge_weight)

    elif backend == "sparse":
        N = adata.shape[0]
        edge_index = adata.uns["graph"]["edge_index"]

        A = scipy.sparse.csr_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(N, N)
        )
        if aggr == "mean":
            A = A / A.sum(1)

        for _ in range(n_layers):
            X = A @ X

    if out_key is not None:
        adata.obsm[out_key] = to_numpy(X)
    elif obsm_key is None:
        adata.X = X if backend == "sparse" else to_numpy(X)
    else:
        adata.obsm[obsm_key + suffix] = to_numpy(X)
