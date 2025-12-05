from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from anndata import AnnData


def leiden_fn(adata: AnnData, resolution: float, **kwargs):
    """
    Leiden function for parallel processing.

    Parameters
    ----------
    adata : AnnData
        Anndata object with the graph information.
    resolution : float
        Resolution parameter for the Leiden algorithm.
    **kwargs : dict
        Additional keyword arguments to be passed to sc.tl.leiden.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with the cluster assignments for each cell.
    """
    flavor = kwargs.pop("flavor", "igraph")
    n_iterations = kwargs.pop("n_iterations", 2 if flavor == "igraph" else -1)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="leiden",
        flavor=flavor,
        n_iterations=n_iterations,
        **kwargs,
    )
    return adata.obs[["leiden"]]


def leiden_multires(
    adata: AnnData,
    resolutions: list[float],
    parallel=True,
    n_jobs=-1,
    prefix="leiden_",
    return_leiden=True,
    **kwargs,
):
    """
    Perform Leiden clustering at multiple resolutions in parallel.

    Parameters
    ----------
    adata : AnnData
        Anndata object with the graph information.
    resolutions : list[float]
        List of resolution parameters for the Leiden algorithm.
    parallel : bool, optional
        Whether to perform the clustering in parallel, by default True.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1.
    prefix : str, optional
        Prefix to be added to the column names, by default ``"leiden_"``.
    return_leiden : bool, optional
        Whether to return the cluster assignments as a DataFrame, by default True.

    Returns
    -------
    pd.DataFrame | None
        Pandas DataFrame with the cluster assignments for each cell.
        If return_leiden is False, the cluster assignments are added to the adata object.
    """
    # create simplified AnnData object with only the graph information
    ad_tmp = sc.AnnData(
        X=csr_matrix((adata.shape[0], 0), dtype=int),
        obsp={
            "connectivities": adata.obsp["connectivities"],
            "distances": adata.obsp["distances"],
        },
        uns={"neighbors": adata.uns["neighbors"]},
    )
    # Execute the clustering in parallel using joblib
    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(leiden_fn)(ad_tmp, res, **kwargs) for res in tqdm(resolutions)
        )
    else:
        results = [leiden_fn(ad_tmp, res, **kwargs) for res in tqdm(resolutions)]

    # convert to dataframe
    results = pd.concat(results, axis=1)
    results.columns = [f"{prefix}{res}" for res in resolutions]

    if return_leiden:
        return results
    else:
        adata.obs = pd.concat([adata.obs, results], axis=1)


def leiden_with_nclusters(
    adata: AnnData,
    n_clusters: int,
    max_res: float = 2.0,
    max_iter: int = 20,
    seed: int = 42,
    min_res: float = 0.0,
    min_cells: int | None = None,
    verbose: bool = False,
    key_added="leiden",
    **kwargs,
):
    """
    Perform Leiden clustering with a fixed number of clusters.

    Parameters
    ----------
    adata : AnnData
        Anndata object with the graph information.
    n_clusters : int
        Number of clusters to be identified.
    max_res : float, optional
        Maximum resolution parameter for the Leiden algorithm, by default 2.0.
    max_iter : int, optional
        Maximum number of iterations, by default 20.
    seed : int, optional
        Random seed for reproducibility, by default 42.
    min_res : float, optional
        Minimum resolution parameter for the Leiden algorithm, by default 0.0.
    min_cells : int | None, optional
        Minimum number of cells per cluster, by default None.
    verbose : bool, optional
        Whether to print information about the clustering process, by default False.
    key_added : str, optional
        Key to store the cluster assignments in the adata object, by default "leiden".
    **kwargs : dict
        Additional keyword arguments to be passed to sc.tl.leiden.

    Returns
    -------
    None
    """
    # Initialize a random state with a fixed seed
    rng = np.random.RandomState(seed=seed)

    cur_res = rng.uniform(min_res, max_res)

    flavor = kwargs.pop("flavor", "igraph")
    n_iterations = kwargs.pop("n_iterations", 2 if flavor == "igraph" else -1)

    for i in range(max_iter):
        sc.tl.leiden(
            adata,
            resolution=cur_res,
            key_added=key_added,
            flavor=flavor,
            n_iterations=n_iterations,
            **kwargs,
        )

        # if min_cells provided counts only clusters with more than min_cells
        cluster_counts = adata.obs[key_added].value_counts()
        if min_cells is None:
            cur_n_clusters = cluster_counts.shape[0]
        else:
            cur_n_clusters = cluster_counts[cluster_counts > min_cells].shape[0]

        if verbose:
            print(f"Resolution {cur_res} has {cur_n_clusters} clusters")

        if cur_n_clusters < n_clusters:
            min_res = cur_res
        elif cur_n_clusters > n_clusters:
            max_res = cur_res
        else:
            if verbose:
                print(
                    f"Found resolution {cur_res} with {n_clusters} clusters after {i + 1} iterations"
                )
            break

        # Update to the mid point
        cur_res = (min_res + max_res) / 2


def leiden_unique(
    adata: AnnData,
    use_rep: str | None = None,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    key_added: str = "leiden",
    flavor: str = "igraph",
    n_iterations: int = 2,
    **kwargs,
):
    """
    Perform Leiden clustering with duplicate embeddings.

    Parameters
    ----------
    adata : AnnData
        Anndata object with the graph information.
    use_rep : str, optional
        The embedding to use for clustering, by default None.
    resolution : float, optional
        The resolution parameter for the Leiden algorithm, by default 1.0.
    n_neighbors : int, optional
        The number of neighbors to use for the Leiden algorithm, by default 15.
    key_added : str, optional
        The key to store the cluster assignments in the adata object, by default "leiden".
    flavor : str, optional
        The flavor of the Leiden algorithm to use, by default "igraph".
    n_iterations : int, optional
        The number of iterations for the Leiden algorithm, by default 2.
    **kwargs : dict
        Additional keyword arguments to be passed to sc.tl.leiden.

    Returns
    -------
    None
    """
    X_rep = adata.obsm[use_rep]
    _, unique_indices, inverse_indices = np.unique(
        X_rep, axis=0, return_index=True, return_inverse=True
    )
    print(f"Found {len(unique_indices)} unique embeddings from a total of {len(X_rep)}")

    ad_sub = adata[unique_indices].copy()
    sc.pp.neighbors(ad_sub, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.leiden(
        ad_sub,
        resolution=resolution,
        flavor=flavor,
        n_iterations=n_iterations,
        key_added=key_added,
        **kwargs,
    )
    labels = ad_sub.obs[key_added].iloc[inverse_indices].copy()
    adata.obs[key_added] = labels.values
