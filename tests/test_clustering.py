import numpy as np
import pandas as pd
import scanpy as sc
from utils import generate_dummy_adata

import nichepca as npc


def test_leiden_multires():
    adata = generate_dummy_adata()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")

    resolutions = np.linspace(0.1, 1.0, 2)

    results_parallel = npc.cl.leiden_multires(
        adata, resolutions, parallel=True, n_jobs=2
    )
    results_sequential = npc.cl.leiden_multires(adata, resolutions, parallel=False)

    assert isinstance(results_parallel, pd.DataFrame)
    assert results_parallel.shape == (adata.shape[0], len(resolutions))
    assert results_parallel.equals(results_sequential)

    npc.cl.leiden_multires(adata, resolutions, parallel=False, return_leiden=False)

    assert f"leiden_{resolutions[0]}" in adata.obs.columns
    assert f"leiden_{resolutions[1]}" in adata.obs.columns


def test_leiden_with_nclusters():
    adata = generate_dummy_adata()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")

    n_clusters = 5
    npc.cl.leiden_with_nclusters(adata, n_clusters)

    assert adata.obs["leiden"].nunique() == n_clusters
