from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc

from nichepca.graph_construction import (
    construct_multi_sample_graph,
    resolve_graph_constructor,
)
from nichepca.nhood_embedding import aggregate
from nichepca.utils import check_for_raw_counts, normalize_per_sample

if TYPE_CHECKING:
    from anndata import AnnData


def nichepca(
    adata: AnnData,
    knn: int | None = None,
    radius: float | None = None,
    delaunay: bool = False,
    n_comps: int = 30,
    obs_key: str | None = None,
    obsm_key: str | None = None,
    sample_key: str | None = None,
    pipeline: tuple | list = ("norm", "log1p", "agg", "pca"),
    norm_per_sample: bool = True,
    backend: str = "pyg",
    aggr: str = "mean",
    allow_harmony: bool = True,
    max_iter_harmony: int = 50,
    **kwargs,
):
    """
    Run the general NichePCA workflow.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object.
    knn : int | None, optional
        Number of nearest neighbors to use for graph construction.
    radius : float | None, optional
        Radius for graph construction.
    delaunay : bool, optional
        Whether to use Delaunay triangulation for graph construction.
    n_comps : int, optional
        Number of principal components to compute.
    obs_key : str | None, optional
        Observation key to use for generating a new AnnData object.
    obsm_key : str | None, optional
        Observation matrix key to use as input.
    sample_key : str | None, optional
        Sample key to use for multi-sample graph construction.
    pipeline : tuple | list, optional
        Pipeline of steps to perform. Must include 'agg'.
    norm_per_sample : bool, optional
        Whether to normalize per sample.
    backend : str, optional
        Backend to use for aggregation.
    aggr : str, optional
        Aggregation method to use.
    allow_harmony : bool, optional
        Whether to allow Harmony integration.
    max_iter_harmony : int, optional
        Maximum number of iterations for Harmony.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    None
    """
    # we always need to use agg
    assert "agg" in pipeline, "aggregation must be part of the pipeline"
    # assert that the pca is behind norm and log1p
    if "pca" in pipeline:
        pca_after_norm = np.argmax(np.array(pipeline) == "pca") > np.argmax(
            np.array(pipeline) == "norm"
        )
        pca_after_log1p = np.argmax(np.array(pipeline) == "pca") > np.argmax(
            np.array(pipeline) == "log1p"
        )
        assert (
            pca_after_norm and pca_after_log1p
        ), "pca must be executed after norm and log1p"

    # perform sanity check in case we are normalizing the data
    if "norm" or "log1p" in pipeline and obs_key is None and obsm_key is None:
        check_for_raw_counts(adata)

    # extract any additional kwargs that are not directed to the graph construction
    target_sum = kwargs.pop("target_sum", None)

    # construct the (multi-sample) graph
    if sample_key is not None:
        construct_multi_sample_graph(
            adata,
            sample_key=sample_key,
            knn=knn,
            radius=radius,
            delaunay=delaunay,
            **kwargs,
        )
    else:
        resolve_graph_constructor(radius, knn, delaunay)(adata, **kwargs)

    # if an obs_key is provided generate a new AnnData
    if obs_key is not None:
        df = pd.get_dummies(adata.obs[obs_key], dtype=np.int8)
        X = df.values
        var = pd.DataFrame(index=df.columns)
        # remove normalization steps
        pipeline = [p for p in pipeline if p not in ["norm", "log1p"]]
        print(f"obs_key provided, running pipeline: {'->'.join(pipeline)}")
    elif obsm_key is not None:
        X = adata.obsm[obsm_key]
        var = adata.var[[]]
    else:
        X = adata.X
        var = adata.var[[]]
        print(f"Running pipeline: {'->'.join(pipeline)}")

    # create intermediate AnnData
    ad_tmp = sc.AnnData(
        X=X,
        obs=adata.obs,
        var=var,
        uns=adata.uns,
    )

    for fn in pipeline:
        if fn == "norm":
            if norm_per_sample and sample_key is not None:
                normalize_per_sample(
                    ad_tmp, sample_key=sample_key, target_sum=target_sum
                )
            else:
                sc.pp.normalize_total(ad_tmp, target_sum=target_sum)
        elif fn == "log1p":
            sc.pp.log1p(ad_tmp)
        elif fn == "agg":
            # if pca is executed before agg, we need to aggregate the pca results
            if "X_pca_harmony" in ad_tmp.obsm:
                obsm_key_agg = "X_pca_harmony"
            elif "X_pca" in ad_tmp.obsm:
                obsm_key_agg = "X_pca"
            else:
                obsm_key_agg = None
            aggregate(
                ad_tmp,
                backend=backend,
                aggr=aggr,
                obsm_key=obsm_key_agg,
                suffix="",
            )
        elif fn == "pca":
            sc.tl.pca(ad_tmp, n_comps=n_comps)
            # run harmony if sample_key is provided and obs key is None
            if sample_key is not None and obs_key is None and allow_harmony:
                sc.external.pp.harmony_integrate(
                    ad_tmp, key=sample_key, max_iter_harmony=max_iter_harmony
                )
        else:
            raise ValueError(f"Unknown step in the pipeline: {fn}")

    # extract the results and remove old keys
    if "X_pca_harmony" in ad_tmp.obsm:
        X_npca = ad_tmp.obsm["X_pca_harmony"]
    else:
        X_npca = ad_tmp.obsm["X_pca"]

    # store the results
    adata.obsm["X_npca"] = X_npca
    adata.uns["npca"] = ad_tmp.uns["pca"]
    adata.uns["npca"]["PCs"] = pd.DataFrame(
        data=ad_tmp.varm["PCs"],
        index=ad_tmp.var_names,
        columns=[f"PC{i}" for i in range(n_comps)],
    )
