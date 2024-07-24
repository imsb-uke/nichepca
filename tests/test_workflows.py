import numpy as np
import scanpy as sc
from utils import generate_dummy_adata

import nichepca as npc


def test_nichepca_single():
    adata_1 = generate_dummy_adata()
    npc.wf.run_nichepca(adata_1, knn=10, n_comps=30)

    adata_2 = generate_dummy_adata()
    sc.pp.normalize_total(adata_2)
    sc.pp.log1p(adata_2)
    npc.gc.knn_graph(adata_2, knn=10)
    npc.ne.aggregate(adata_2)
    sc.tl.pca(adata_2, n_comps=30)

    assert np.all(adata_1.obsm["X_pca"] == adata_2.obsm["X_pca"])


def test_nichepca_multi():
    adata = generate_dummy_adata()
    npc.wf.run_nichepca(adata, knn=10, sample_key="sample")

    assert "X_pca" in adata.obsm.keys()
    assert "X_pca_harmony" in adata.obsm.keys()
