# NichePCA: PCA-based spatial domain identification with state-of-the-art performance

[![Version](https://img.shields.io/pypi/v/nichepca)](https://pypi.org/project/nichepca/)
[![License](https://img.shields.io/pypi/l/nichepca)](https://github.com/imsb-uke/nichepca)
[![Python Version Required](https://img.shields.io/pypi/pyversions/nichepca)](https://pypi.org/project/nichepca/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPI downloads](https://img.shields.io/pepy/dt/nichepca?label=PyPI%20downloads&logo=pypi)](https://pepy.tech/project/nichepca)
[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/imsb-uke/nichepca/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/nichepca

NichePCA is a package for PCA-based spatial domain identification in single-cell spatial transcriptomics data. The corresponding manuscript was published in [Bioinformatics](https://academic.oup.com/bioinformatics/article/41/1/btaf005/7945104?).

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install nichepca:


1. Install the latest release of `nichepca` from [PyPI][]:

```bash
pip install nichepca
```


2. Install the latest development version:

```bash
pip install git+https://github.com/imsb-uke/nichepca.git@main
```

## Getting started

Please refer to the [documentation][]. In particular, the [API documentation][].

Given an AnnData object `adata`, you can run nichepca starting from raw counts as follows:

```python
import scanpy as sc
import nichepca as npc

npc.wf.nichepca(adata, knn=25)
sc.pp.neighbors(adata, use_rep="X_npca")
sc.tl.leiden(adata, resolution=0.5)
```
### Multi-sample support

If you have multiple samples in `adata.obs["sample"]`, you can provide the key `sample` to `npc.wf.nichepca` this uses harmony by default:

```python
npc.wf.nichepca(adata, knn=25, sample_key="sample")
```

If you have cell type labels in `adata.obs["cell_type"]`, you can directly provide them to `nichepca` as follows (we found this sometimes works better for multi-sample domain identification). However, in this case we need to run `npc.cl.leiden_unique` to handle potential duplicate embeddings:

```python
npc.wf.nichepca(adata, knn=25, obs_key="cell_type", sample_key="sample")
npc.cl.leiden_unique(adata, use_rep="X_npca", resolution=0.5, n_neighbors=15)
```

### Customization

The `nichepca` function also allows to customize the original `("norm", "log1p", "agg", "pca")` pipeline, e.g., without median normalization:
```python
npc.wf.nichepca(adata, knn=25, pipeline=["log1p", "agg", "pca"])
```
or with `"pca"` before `"agg"`:
```python
npc.wf.nichepca(adata, knn=25, pipeline=["norm", "log1p", "pca", "agg"])
```
or without `"pca"` at all:
```python
npc.wf.nichepca(adata, knn=25, pipeline=["norm", "log1p", "agg"])
```

## Hyperparameter choice
We found that higher number of neighbors e.g., `knn=25` lead to better results in brain tissue, while `knn=10` works well for kidney data. We recommend to qualitatively optimize these parameters on a small subset of your data. The number of PCs (`n_comps=30` by default) seems to have negligible effect on the results.

## Contributing

If you want to contribute you can follow this [guide](https://scanpy.readthedocs.io/en/latest/dev/index.html). In short fork the repository, setup a dev environment using this command:

```bash
git clone https://github.com/{your-username}/nichepca.git
cd nichepca
uv sync --all-extras
```
And then make your changes, run the tests and submit a pull request.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

If you use NichePCA in your research, please cite:
```
@article{schaub2025pca,
  title={PCA-based spatial domain identification with state-of-the-art performance},
  author={Schaub, Darius P and Yousefi, Behnam and Kaiser, Nico and Khatri, Robin and Puelles, Victor G and Krebs, Christian F and Panzer, Ulf and Bonn, Stefan},
  journal={Bioinformatics},
  volume={41},
  number={1},
  pages={btaf005},
  year={2025},
  publisher={Oxford University Press}
}
```
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/imsb-uke/nichepca/issues
[tests]: https://github.com/imsb-uke/nichepca/actions/workflows/test.yaml
[documentation]: https://nichepca.readthedocs.io
[changelog]: https://nichepca.readthedocs.io/en/latest/changelog.html
[api documentation]: https://nichepca.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/nichepca
