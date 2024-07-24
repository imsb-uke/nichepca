import numpy as np
from utils import generate_dummy_adata

import nichepca as npc


def test_aggregate():
    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=10)
    npc.ne.aggregate(adata, n_layers=1, backend="pyg")
    X_agg_1 = npc.utils.to_numpy(adata.X)

    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=10)
    npc.ne.aggregate(adata, n_layers=1, backend="sparse")
    X_agg_2 = npc.utils.to_numpy(adata.X)

    assert np.allclose(X_agg_1, X_agg_2, atol=1e-7)

    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=10)
    npc.ne.aggregate(adata, n_layers=2, backend="pyg")
    X_agg_1 = npc.utils.to_numpy(adata.X)

    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=10)
    npc.ne.aggregate(adata, n_layers=2, backend="sparse")
    X_agg_2 = npc.utils.to_numpy(adata.X)

    assert np.allclose(X_agg_1, X_agg_2, atol=1e-7)
