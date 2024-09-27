import contextlib
import io

import numpy as np
import pytest
import torch
from utils import generate_dummy_adata

import nichepca as npc


def test_distance_graph():
    adata = generate_dummy_adata()
    npc.gc.distance_graph(adata, radius=0.1)

    assert "graph" in adata.uns
    assert "edge_index" in adata.uns["graph"]
    assert "edge_weight" in adata.uns["graph"]

    (
        num_nodes,
        num_edges,
        in_degrees,
        out_degrees,
        contains_self_loops,
        is_undirected,
    ) = npc.gc.calc_graph_stats(adata=adata)

    assert num_nodes == adata.n_obs
    assert num_edges > 0

    assert out_degrees.mean() == in_degrees.mean()

    assert contains_self_loops
    assert is_undirected


def test_knn_graph():
    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=10)

    assert "graph" in adata.uns
    assert "edge_index" in adata.uns["graph"]
    assert "edge_weight" in adata.uns["graph"]

    (
        num_nodes,
        num_edges,
        in_degrees,
        out_degrees,
        contains_self_loops,
        is_undirected,
    ) = npc.gc.calc_graph_stats(adata=adata)

    assert num_nodes == adata.n_obs
    assert num_edges > 0

    assert in_degrees.mean() >= 11
    assert out_degrees.mean() == in_degrees.mean()

    assert contains_self_loops
    assert is_undirected

    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=10, undirected=False, remove_self_loops=True)

    (
        num_nodes,
        num_edges,
        in_degrees,
        out_degrees,
        contains_self_loops,
        is_undirected,
    ) = npc.gc.calc_graph_stats(adata=adata)

    assert not contains_self_loops
    assert not is_undirected


def test_construct_multi_sample_graph():
    adata_1 = generate_dummy_adata(n_cells=50)
    adata_2 = generate_dummy_adata(n_cells=50)

    sample_key = "sample"

    # standard multi sample variant
    npc.gc.construct_multi_sample_graph(
        adata_1, knn=3, sample_key=sample_key, verbose=False, keep_local_edge_index=True
    )

    adata_2.uns["graph"] = {}
    for sample in adata_2.obs[sample_key].unique():
        mask = adata_2.obs[sample_key] == sample
        sub_ad = adata_2[mask].copy()
        del sub_ad.uns
        npc.gc.knn_graph(sub_ad, knn=3, verbose=False)
        adata_2.uns["graph"][sample] = sub_ad.uns["graph"].copy()

    for sample in adata_2.obs[sample_key].unique():
        assert np.all(
            adata_1.uns["graph"][sample]["edge_index"]
            == adata_2.uns["graph"][sample]["edge_index"]
        )

    # test via multi-sample aggregation
    adata_1 = generate_dummy_adata(n_cells=50)
    adata_2 = generate_dummy_adata(n_cells=50)
    adata_3 = generate_dummy_adata(n_cells=50)

    # standard multi sample variant
    npc.gc.construct_multi_sample_graph(
        adata_1, knn=3, sample_key=sample_key, verbose=False
    )
    npc.ne.aggregate(adata_1)

    # extract subgraph info
    npc.gc.construct_multi_sample_graph(
        adata_2, knn=3, sample_key=sample_key, verbose=False, keep_local_edge_index=True
    )
    # need to convert to float otherwise setting the values will convert to int
    adata_2.X = npc.utils.to_numpy(adata_2.X).astype(np.float32)
    for batch in adata_2.obs[sample_key].unique():
        mask = adata_2.obs[sample_key] == batch
        sub_ad = adata_2[mask].copy()
        sub_ad.uns["graph"] = adata_2.uns["graph"][batch].copy()
        npc.ne.aggregate(sub_ad)
        adata_2.X[mask] = sub_ad.X.copy()

    # manual variant
    adata_3.X = npc.utils.to_numpy(adata_3.X).astype(np.float32)
    for batch in adata_3.obs[sample_key].unique():
        mask = adata_3.obs[sample_key] == batch
        sub_ad = adata_3[mask].copy()
        npc.gc.knn_graph(sub_ad, knn=3, verbose=False)
        npc.ne.aggregate(sub_ad)
        adata_3.X[mask] = sub_ad.X.copy()

    assert (adata_1.X == adata_2.X).all()
    assert (adata_1.X == adata_3.X).all()


def test_print_graph_stats():
    adata = generate_dummy_adata()
    npc.gc.knn_graph(adata, knn=10)

    with contextlib.redirect_stdout(io.StringIO()) as f:
        npc.gc.print_graph_stats(adata=adata)
        assert len(f.getvalue()) > 0

    with pytest.raises(ValueError) as e:
        npc.gc.print_graph_stats()
    assert str(e.value) == "Either adata or edge_index must be provided."


def test_squidpy_conversion():
    adata = generate_dummy_adata()

    npc.gc.knn_graph(adata, knn=10)
    edge_index = adata.uns["graph"]["edge_index"].copy()
    edge_weight = adata.uns["graph"]["edge_weight"].copy()

    # squidpy conversion
    npc.gc.to_squidpy(adata)
    npc.gc.from_squidpy(adata)
    edge_index_new = adata.uns["graph"]["edge_index"].copy()
    edge_weight_new = adata.uns["graph"]["edge_weight"].copy()

    assert np.all(edge_index == edge_index_new)
    assert np.all(edge_weight == edge_weight_new)


def test_resolve_graph_constructor():
    adata = generate_dummy_adata()

    knn = 10
    edge_index_1, edge_weight_1 = npc.gc.knn_graph(adata, knn=knn, return_graph=True)
    edge_index_2, edge_weight_2 = npc.gc.resolve_graph_constructor(knn=knn)(
        adata, return_graph=True
    )

    assert torch.all(edge_index_1 == edge_index_2)
    assert torch.all(edge_weight_1 == edge_weight_2)

    radius = 0.1
    edge_index_1, edge_weight_1 = npc.gc.distance_graph(
        adata, radius=radius, return_graph=True
    )
    edge_index_2, edge_weight_2 = npc.gc.resolve_graph_constructor(radius=radius)(
        adata, return_graph=True
    )

    assert torch.all(edge_index_1 == edge_index_2)
    assert torch.all(edge_weight_1 == edge_weight_2)

    edge_index_1, edge_weight_1 = npc.gc.delaunay_graph(adata, return_graph=True)
    edge_index_2, edge_weight_2 = npc.gc.resolve_graph_constructor(delaunay=True)(
        adata, return_graph=True
    )
    print(type(edge_index_1), type(edge_index_2))

    assert torch.all(edge_index_1 == edge_index_2)
    assert torch.all(edge_weight_1 == edge_weight_2)
