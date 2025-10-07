import numpy as np
import pandas as pd
import scanpy as sc
from utils import generate_dummy_adata

import nichepca as npc


def test_nichepca_single():
    adata_1 = generate_dummy_adata()
    npc.wf.nichepca(adata_1, knn=10, n_comps=30)

    adata_2 = generate_dummy_adata()
    sc.pp.normalize_total(adata_2)
    sc.pp.log1p(adata_2)
    npc.gc.knn_graph(adata_2, knn=10)
    npc.ne.aggregate(adata_2)
    sc.tl.pca(adata_2, n_comps=30)

    assert np.all(adata_1.obsm["X_npca"] == adata_2.obsm["X_pca"])

    # test with obs_key
    obs_key = "cell_type"
    n_celltypes = 5
    adata_1 = generate_dummy_adata(n_celltypes=n_celltypes)
    npc.wf.nichepca(adata_1, knn=10, n_comps=n_celltypes - 1, obs_key=obs_key)

    adata_2 = generate_dummy_adata(n_celltypes=n_celltypes)
    npc.gc.knn_graph(adata_2, knn=10)
    df = pd.get_dummies(adata_2.obs[obs_key], dtype=np.int8)
    ad_tmp = sc.AnnData(
        X=df.values,
        obs=adata_2.obs,
        var=pd.DataFrame(index=df.columns),
        uns=adata_2.uns,
    )
    npc.ne.aggregate(ad_tmp)
    sc.tl.pca(ad_tmp, n_comps=n_celltypes - 1)

    assert np.all(adata_1.obsm["X_npca"] == ad_tmp.obsm["X_pca"])

    # test with pca before agg
    adata_1 = generate_dummy_adata()
    npc.wf.nichepca(
        adata_1, knn=10, n_comps=30, pipeline=["norm", "log1p", "pca", "agg"]
    )

    adata_2 = generate_dummy_adata()
    npc.gc.knn_graph(adata_2, knn=10)
    sc.pp.normalize_total(adata_2)
    sc.pp.log1p(adata_2)
    sc.pp.pca(adata_2, n_comps=30)
    npc.ne.aggregate(adata_2, obsm_key="X_pca", suffix="")

    assert np.all(adata_1.obsm["X_npca"] == adata_2.obsm["X_pca"])

    # test without pca
    pipeline = "agg"

    adata = generate_dummy_adata()
    npc.wf.nichepca(adata, knn=5, pipeline=pipeline)
    X_npca_0 = adata.obsm["X_npca"]

    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=5)
    npc.ne.aggregate(adata)
    X_npca_1 = npc.utils.to_numpy(adata.X)

    assert np.all(X_npca_0 == X_npca_1)

    # test graph removal
    adata = generate_dummy_adata()
    npc.wf.nichepca(adata, knn=5, pipeline=pipeline, remove_graph=True)
    assert "graph" not in adata.uns

    # test with pca on raw counts
    pipeline = ("pca", "agg")

    adata = generate_dummy_adata()
    npc.wf.nichepca(adata, knn=5, pipeline=pipeline)
    X_npca_0 = adata.obsm["X_npca"]

    adata = generate_dummy_adata()
    adata.X = adata.X.astype(np.float32)
    sc.pp.pca(adata, n_comps=30)
    npc.gc.knn_graph(adata, knn=5)
    npc.ne.aggregate(adata, obsm_key="X_pca")
    X_npca_1 = adata.obsm["X_pca_agg"]

    assert np.all(X_npca_0 == X_npca_1)


def test_nichepca_multi_sample():
    adata_1 = generate_dummy_adata()
    npc.wf.nichepca(adata_1, knn=10, n_comps=30, sample_key="sample")

    adata_2 = generate_dummy_adata()
    npc.gc.construct_multi_sample_graph(adata_2, knn=10, sample_key="sample")
    npc.utils.normalize_per_sample(adata_2, sample_key="sample")
    sc.pp.log1p(adata_2)
    npc.ne.aggregate(adata_2)
    sc.tl.pca(adata_2, n_comps=30)
    sc.external.pp.harmony_integrate(adata_2, key="sample", max_iter_harmony=50)

    assert np.all(adata_1.obsm["X_npca"] == adata_2.obsm["X_pca_harmony"])

    # test with obs_key
    obs_key = "cell_type"
    n_celltypes = 5
    adata_1 = generate_dummy_adata(n_celltypes=n_celltypes)
    npc.wf.nichepca(
        adata_1, knn=10, n_comps=n_celltypes - 1, obs_key=obs_key, sample_key="sample"
    )

    adata_2 = generate_dummy_adata(n_celltypes=n_celltypes)
    npc.gc.construct_multi_sample_graph(adata_2, knn=10, sample_key="sample")
    df = pd.get_dummies(adata_2.obs[obs_key], dtype=np.int8)
    ad_tmp = sc.AnnData(
        X=df.values,
        obs=adata_2.obs,
        var=pd.DataFrame(index=df.columns),
        uns=adata_2.uns,
    )
    npc.ne.aggregate(ad_tmp)
    sc.tl.pca(ad_tmp, n_comps=n_celltypes - 1)

    assert np.all(adata_1.obsm["X_npca"] == ad_tmp.obsm["X_pca"])
