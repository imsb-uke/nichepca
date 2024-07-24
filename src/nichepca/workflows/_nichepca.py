from __future__ import annotations

from typing import TYPE_CHECKING

import scanpy as sc

from nichepca.graph_construction import (
    construct_multi_sample_graph,
    distance_graph,
    knn_graph,
)
from nichepca.nhood_embedding import aggregate
from nichepca.utils import check_for_raw_counts

if TYPE_CHECKING:
    from anndata import AnnData


def run_nichepca(
    adata: AnnData,
    knn: int = None,
    radius: float = None,
    sample_key: str = None,
    n_comps: int = 30,
    max_iter_harmony: int = 50,
    norm_per_sample: bool = True,
    **kwargs,
):
    """
    Run the NichePCA workflow.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    knn : int
        Number of nearest neighbors for the kNN graph.
    sample_key : str, optional
        Key in `adata.obs` that identifies distinct samples. If provided, harmony will be used to
        integrate the data.
    radius : float, optional
        The radius of the neighborhood graph.
    n_comps : int, optional
        Number of principal components to compute.
    max_iter_harmony : int, optional
        Maximum number of iterations for harmony.
    norm_per_sample : bool, optional
        Whether to normalize the data per sample.
    kwargs : dict, optional
        Additional keyword arguments for the graph construction.

    Returns
    -------
    None
    """
    check_for_raw_counts(adata)

    if sample_key is not None:
        construct_multi_sample_graph(
            adata, sample_key=sample_key, knn=knn, radius=radius, **kwargs
        )
    else:
        if knn is not None:
            knn_graph(adata, knn, **kwargs)
        elif radius is not None:
            distance_graph(adata, radius, **kwargs)
        else:
            raise ValueError("Either knn or radius must be provided.")

    if norm_per_sample and sample_key is not None:
        for sample in adata.obs[sample_key].unique():
            mask = adata.obs[sample_key] == sample
            sub_ad = adata[mask].copy()
            sc.pp.normalize_total(sub_ad)
            adata[mask].X = sub_ad.X
    else:
        sc.pp.normalize_total(adata)

    sc.pp.log1p(adata)

    aggregate(adata)

    sc.tl.pca(adata, n_comps=n_comps)

    if sample_key is not None:
        sc.external.pp.harmony_integrate(
            adata, key=sample_key, max_iter_harmony=max_iter_harmony
        )
