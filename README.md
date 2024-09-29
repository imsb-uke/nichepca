# NichePCA

[![Tests][badge-tests]][link-tests]
<!-- [![Documentation][badge-docs]][link-docs] -->

[badge-tests]: https://img.shields.io/github/actions/workflow/status/imsb-uke/nichepca/test.yaml?branch=main
[link-tests]: https://github.com/imsb-uke/nichepca/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/nichepca

Package for PCA-based spatial domain identification in single-cell spatial transcriptomics data

## Getting started

<!-- Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api]. -->

Given an AnnData object `adata`, you can run nichepca starting from raw counts as follows:

```python
import scanpy as sc
import nichepca as npc

npc.wf.nichepca(adata, knn=25)
sc.pp.neighbors(adata, use_rep="X_npca")
sc.tl.leiden(adata, resolution=0.5)
```

If you have multiple samples in `adata.obs['sample']`, you can provide the key `sample` to `npc.wf.nichepca`:

```python
npc.wf.nichepca(adata, knn=25, sample_key="sample")
```

If you have cell type labels in `adata.obs['cell_type']`, you can directly provide them to `nichepca` as follows:

```python
npc.wf.nichepca(adata, knn=25, obs_key='cell_type')
```

The `nichepca` functiopn also allows to customize the original `("norm", "log1p", "agg", "pca")` pipeline, e.g., without median normalization:
```python
npc.wf.nichepca(adata, knn=25, pipeline=["log1p", "agg", "pca"])
```

We found that higher number of neighbors e.g., `knn=25` lead to better results in brain tissue, while `knn=10` works well for kidney data. We recommend to qualitatively optimize these parameters on a small subset of your data.

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) or [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).

To create a new conda environment with Python 3.10:

```bash
conda create -n npc-env python=3.10 -y
conda activate npc-env
```

There are several options to install nichepca:

<!--
1) Install the latest release of `nichepca` from [PyPI][link-pypi]:

```bash
pip install nichepca
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/imsb-uke/nichepca.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/imsb-uke/nichepca/issues
[changelog]: https://nichepca.readthedocs.io/latest/changelog.html
[link-docs]: https://nichepca.readthedocs.io
[link-api]: https://nichepca.readthedocs.io/latest/api.html
[link-pypi]: https://pypi.org/project/nichepca
