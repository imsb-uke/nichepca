from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import scipy
import torch
import torch_geometric as pyg
from tqdm.auto import tqdm

from nichepca.utils import to_numpy, to_torch

if TYPE_CHECKING:
    from anndata import AnnData


def calc_graph_stats(
    adata: AnnData | None = None,
    edge_index: torch.tensor | np.ndarray | None = None,
    num_nodes: int | None = None,
):
    """
    Calculate basic graph statistics.

    Parameters
    ----------
    adata : AnnData, optional
        Annotated data object.
    edge_index : Union[torch.Tensor, np.ndarray], optional
        Edge index of the graph.
    num_nodes : int, optional
        Number of nodes in the graph.

    Returns
    -------
    Tuple[int, int, torch.Tensor, torch.Tensor, bool, bool]
    """
    if adata is not None:
        edge_index = torch.from_numpy(adata.uns["graph"]["edge_index"])
        num_nodes = adata.shape[0]
    elif edge_index is not None:
        edge_index = to_torch(edge_index)
    else:
        raise ValueError("Either adata or edge_index must be provided.")

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    num_edges = edge_index.shape[1]
    in_degrees = pyg.utils.degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
    out_degrees = pyg.utils.degree(
        edge_index[0], num_nodes=num_nodes, dtype=torch.float
    )
    contains_self_loops = pyg.utils.contains_self_loops(edge_index)
    is_undirected = pyg.utils.is_undirected(edge_index)
    return (
        num_nodes,
        num_edges,
        in_degrees,
        out_degrees,
        contains_self_loops,
        is_undirected,
    )


def print_graph_stats(
    adata: AnnData | None = None,
    edge_index: torch.tensor | None = None,
    num_nodes: int | None = None,
):
    """
    Print statistics about the graph.

    Parameters
    ----------
    adata : AnnData, optional
        Annotated data object.
    edge_index : torch.Tensor, optional
        Edge index of the graph.
    num_nodes : int, optional
        Number of nodes in the graph.

    Returns
    -------
    None
    """
    (
        num_nodes,
        num_edges,
        in_degrees,
        out_degrees,
        contains_self_loops,
        is_undirected,
    ) = calc_graph_stats(adata=adata, edge_index=edge_index, num_nodes=num_nodes)

    print("----------- Graph Stats -----------")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average in-degree: {in_degrees.mean().item()}")
    print(f"Average out-degree: {out_degrees.mean().item()}")
    print(f"Contains self-loops: {contains_self_loops}")
    print(f"Is undirected: {is_undirected}")


def store_graph(
    adata: AnnData,
    edge_index: torch.tensor | np.ndarray,
    edge_weight: torch.tensor | np.ndarray,
    extra_key: str | None = None,
):
    """
    Store graph data in `adata.uns`.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    edge_index : Union[torch.Tensor, np.ndarray]
        Edge index of the graph.
    edge_weight : Union[torch.Tensor, np.ndarray]
        Edge weight of the graph.
    extra_key : str, optional
        Extra key to store the graph data in `adata.uns`.

    Returns
    -------
    None
    """
    if "graph" not in adata.uns:
        adata.uns["graph"] = {}
    if extra_key is not None:
        adata.uns["graph"][extra_key] = {}
        adata.uns["graph"][extra_key]["edge_index"] = to_numpy(edge_index)
        adata.uns["graph"][extra_key]["edge_weight"] = to_numpy(edge_weight)
    else:
        adata.uns["graph"]["edge_index"] = to_numpy(edge_index)
        adata.uns["graph"]["edge_weight"] = to_numpy(edge_weight)


def knn_graph(
    adata: AnnData,
    knn: int,
    obsm_key: str = "spatial",
    undirected: bool = True,
    remove_self_loops: bool = False,
    p: int = 2,
    verbose: bool = True,
    return_graph=False,
):
    """
    Construct a k-nearest neighbors graph.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    knn : int
        Number of nearest neighbors.
    obsm_key : str, default "spatial"
        Key in `obsm` attribute where the spatial data is stored.
    undirected : bool, default True
        Whether to create an undirected graph.
    remove_self_loops : bool, default False
        Whether to remove self-loops.
    p : int, default 2
        The norm to calculate the distance.
    verbose : bool, default True
        Whether to print graph statistics.
    return_graph : bool, default False
        Whether to return the graph instead of storing it in `adata`.

    Returns
    -------
    None
    """
    if obsm_key is not None:
        coords = adata.obsm[obsm_key]
    else:
        coords = to_numpy(adata.X)

    kdtree = scipy.spatial.KDTree(coords)
    distances, indices = kdtree.query(coords, k=knn + 1, p=p)
    edge_index = torch.cat(
        [
            torch.tensor(indices.flatten())[None, :],  # source
            torch.arange(0, coords.shape[0]).repeat_interleave(knn + 1)[
                None, :
            ],  # target
        ],
        axis=0,
    )
    edge_weight = torch.tensor(distances.flatten()).unsqueeze(-1)

    if undirected:
        edge_index, edge_weight = pyg.utils.to_undirected(edge_index, edge_weight)

    if remove_self_loops:
        edge_index, edge_weight = pyg.utils.remove_self_loops(edge_index, edge_weight)

    if verbose:
        print_graph_stats(edge_index=edge_index, num_nodes=adata.n_obs)

    if not return_graph:
        store_graph(adata, edge_index, edge_weight)
    else:
        return edge_index, edge_weight


def distance_graph(
    adata: AnnData,
    radius: float = 50,
    obsm_key: str = "spatial",
    remove_self_loops: bool = False,
    p: int = 2,
    verbose: bool = True,
    return_graph=False,
):
    """
    Construct a spatial graph based on a distance threshold.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    radius : float, default 50
        Radius for the distance threshold.
    obsm_key : str, default "spatial"
        Key in `obsm` attribute where the spatial data is stored.
    remove_self_loops : bool, default False
        Whether to remove self-loops.
    p : int, default 2
        The norm to calculate the distance.
    verbose : bool, default True
        Whether to print graph statistics.
    return_graph : bool, default False
        Whether to return the graph instead of storing it in `adata`.

    Returns
    -------
    None
    """
    if obsm_key is not None:
        coords = adata.obsm[obsm_key]
    else:
        coords = to_numpy(adata.X)
    kdtree = scipy.spatial.KDTree(coords)
    dist_mat = kdtree.sparse_distance_matrix(kdtree, radius, p=p)

    edge_index, edge_weight = pyg.utils.from_scipy_sparse_matrix(dist_mat)
    edge_weight = edge_weight.unsqueeze(-1)

    if remove_self_loops:
        edge_index, edge_weight = pyg.utils.remove_self_loops(edge_index, edge_weight)

    if verbose:
        print_graph_stats(edge_index=edge_index, num_nodes=adata.n_obs)

    if not return_graph:
        store_graph(adata, edge_index, edge_weight)
    else:
        return edge_index, edge_weight


def remove_long_links(
    adata: AnnData | None = None,
    edge_index: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
    dist_percentile: float = 99.0,
    copy: bool = False,
):
    """
    Remove links with a distance larger than a given percentile.

    Parameters
    ----------
    adata : AnnData, optional
        Annotated data object.
    edge_index : np.ndarray, optional
        Edge index of the graph.
    edge_weight : np.ndarray, optional
        Edge weight of the graph.
    dist_percentile : float, default 99.0
        Percentile threshold for the maximum edge weight.
    copy : bool, default False
        Whether to return a copy of the graph.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] | None
        If `copy=True`, returns the edge index and edge weight.
        Otherwise, stores the new edge index and edge weight in `adata.uns`.

    Notes
    -----
    This function is inspired by the `remove_long_links` function in the `cellcharter`
    package https://github.com/CSOgroup/cellcharter.
    """
    if adata is not None:
        edge_index = adata.uns["graph"]["edge_index"]
        edge_weight = adata.uns["graph"]["edge_weight"]
    elif edge_index is not None and edge_weight is not None:
        edge_index = to_numpy(edge_index)
        edge_weight = to_numpy(edge_weight)
        copy = True
    else:
        raise ValueError("Either adata or edge_index and edge_weight must be provided.")

    if copy:
        edge_index, edge_weight = edge_index.copy(), edge_weight.copy()
    threshold = np.percentile(
        np.array(edge_weight[edge_weight != 0]).squeeze(), dist_percentile
    )
    mask = (edge_weight <= threshold).squeeze()
    edge_weight = edge_weight[mask]
    edge_index = edge_index[:, mask]

    if copy:
        return edge_index, edge_weight
    else:
        store_graph(adata, edge_index, edge_weight)


def delaunay_graph(
    adata: AnnData,
    obsm_key: str = "spatial",
    add_self_loops: bool = False,
    remove_long_links_: bool = True,
    verbose: bool = True,
    return_graph=False,
):
    """
    Construct a delaunay graph.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    obsm_key : str, default "spatial"
        Key in `obsm` attribute where the spatial data is stored.
    add_self_loops : bool, default False
        Whether to add self-loops to the graph.
    remove_long_links_ : bool, default True
        Whether to remove long links (edges).
    verbose : bool, default True
        Whether to print graph statistics.
    return_graph : bool, default False
        Whether to return the graph instead of storing it in `adata`.

    Returns
    -------
    None
    """
    coords = adata.obsm[obsm_key]
    N = coords.shape[0]

    # creates a delaunay triangulation graph
    tri = scipy.spatial.Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    adj_mat = scipy.sparse.csr_matrix(
        (np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N)
    )
    edge_index, _ = pyg.utils.from_scipy_sparse_matrix(adj_mat)
    # calc euclidean distance
    edge_weight = torch.tensor(
        np.linalg.norm(coords[edge_index[0]] - coords[edge_index[1]], axis=1)
    )
    edge_weight = edge_weight.unsqueeze(-1)

    if add_self_loops:
        edge_index, edge_weight = pyg.utils.add_self_loops(
            edge_index, edge_weight, fill_value=0
        )

    if remove_long_links_:
        edge_index, edge_weight = remove_long_links(
            edge_index=edge_index, edge_weight=edge_weight
        )
        # convert back to torch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    if verbose:
        print_graph_stats(edge_index=edge_index, num_nodes=adata.n_obs)

    if not return_graph:
        store_graph(adata, edge_index, edge_weight)
    else:
        return edge_index, edge_weight


def construct_multi_sample_graph(
    adata: AnnData,
    sample_key: str,
    knn: int | None = None,
    radius: float | None = None,
    delaunay: bool = False,
    return_graph: bool = False,
    keep_local_edge_index: bool = False,
    verbose: bool = True,
    **kwargs,
):
    """
    Construct a multi-sample graph from AnnData.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    sample_key : str
        Key in `adata.obs` where sample information is stored.
    knn : int | None, optional
        Number of nearest neighbors. Defaults to None.
    radius : float | None, optional
        Radius for the distance graph constructor. Defaults to None.
    delaunay : bool, optional
        Whether to use the delaunay graph constructor. Defaults to False.
    return_graph : bool, optional
        Whether to return the graph instead of storing it in `adata`. Defaults to False.
    keep_local_edge_index : bool, optional
        Whether to keep the local edge index. Defaults to False.
    verbose : bool, optional
        Whether to print graph statistics. Defaults to True.
    **kwargs
        Additional keyword arguments passed to the graph constructor.

    Returns
    -------
    edge_index, edge_weight : torch.Tensor, torch.Tensor
        Edge index and edge weight of the constructed graph if `return_graph` is True.
    """
    edge_index = []
    edge_weight = []
    global_indices = torch.arange(adata.n_obs)

    if "graph" not in adata.uns and not return_graph:
        adata.uns["graph"] = {}

    for sample in tqdm(adata.obs[sample_key].unique()):
        mask = adata.obs[sample_key] == sample
        ad_sub = adata[mask]

        local_global_indices = global_indices[mask.values]

        local_edge_index, local_edge_weight = resolve_graph_constructor(
            radius, knn, delaunay
        )(
            ad_sub,
            return_graph=True,
            verbose=False,
            **kwargs,
        )

        local_global_edge_index = local_global_indices[local_edge_index]

        edge_index.append(local_global_edge_index)
        edge_weight.append(local_edge_weight)

        if not return_graph:
            store_graph(
                adata,
                local_edge_index if keep_local_edge_index else local_global_edge_index,
                local_edge_weight,
                extra_key=sample,
            )

    edge_index = torch.cat(edge_index, axis=1)
    edge_weight = torch.cat(edge_weight, axis=0)

    if verbose:
        print_graph_stats(edge_index=edge_index, num_nodes=adata.n_obs)

    if not return_graph:
        store_graph(adata, edge_index, edge_weight)
    else:
        return edge_index, edge_weight


def resolve_graph_constructor(
    radius: float | None = None, knn: int | None = None, delaunay: bool = False
):
    """
    Resolves and returns the graph constructor based on the provided parameters.

    Parameters
    ----------
    radius
        The radius for the distance graph constructor.
    knn
        The number of nearest neighbors for the knn graph constructor.
    delaunay
        Whether to use the delaunay graph constructor.

    Returns
    -------
    callable
        The resolved graph constructor function with the appropriate parameters.

    Raises
    ------
    ValueError
        If no graph constructor is specified.
    """
    if radius is not None:
        return partial(distance_graph, radius=radius)
    elif knn is not None:
        return partial(knn_graph, knn=knn)
    elif delaunay:
        return delaunay_graph
    else:
        raise ValueError("No graph constructor specified.")


def to_squidpy(adata: AnnData):
    """
    Convert the pyg graph stored in `adata.uns` to squidpy format.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.

    Returns
    -------
    None
    """
    N = adata.shape[0]
    edge_index = adata.uns["graph"]["edge_index"]
    edge_weight = adata.uns["graph"]["edge_weight"]

    row_indices, col_indices = edge_index
    distances = edge_weight.squeeze()

    adj_mat = scipy.sparse.csr_matrix(
        (np.ones_like(distances), (row_indices, col_indices)), shape=(N, N)
    )
    dist_mat = scipy.sparse.csr_matrix(
        (distances, (row_indices, col_indices)), shape=(N, N)
    )
    adata.obsp["spatial_connectivities"] = adj_mat
    adata.obsp["spatial_distances"] = dist_mat


def from_squidpy(adata: AnnData):
    """
    Convert the graph stored in squidpy format to pyg format.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.

    Returns
    -------
    None
    """
    dist_mat = adata.obsp["spatial_distances"]

    edge_index, edge_weight = pyg.utils.from_scipy_sparse_matrix(dist_mat)

    edge_weight = edge_weight.unsqueeze(-1)

    store_graph(adata, edge_index, edge_weight)
