# API

## Workflows

```{eval-rst}
.. autosummary::
    :toctree: generated

    nichepca.workflows
    nichepca.workflows.nichepca
```

## Clustering

```{eval-rst}
.. autosummary::
    :toctree: generated

    nichepca.clustering
    nichepca.clustering.leiden_unique
    nichepca.clustering.leiden_multires
    nichepca.clustering.leiden_with_nclusters
```

## Graph Construction

```{eval-rst}
.. autosummary::
    :toctree: generated

    nichepca.graph_construction
    nichepca.graph_construction.knn_graph
    nichepca.graph_construction.delaunay_graph
    nichepca.graph_construction.distance_graph
    nichepca.graph_construction.construct_multi_sample_graph
    nichepca.graph_construction.from_squidpy
    nichepca.graph_construction.to_squidpy
    nichepca.graph_construction.remove_long_links
    nichepca.graph_construction.calc_graph_stats
    nichepca.graph_construction.print_graph_stats
    nichepca.graph_construction.resolve_graph_constructor
```

## Neighborhood Embedding

```{eval-rst}
.. autosummary::
    :toctree: generated

    nichepca.nhood_embedding
    nichepca.nhood_embedding.aggregate
```

## Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated

    nichepca.utils
    nichepca.utils.check_for_raw_counts
    nichepca.utils.normalize_per_sample
    nichepca.utils.to_numpy
    nichepca.utils.to_torch
```
