# NichePCA

[![Tests][badge-tests]][link-tests]
<!-- [![Documentation][badge-docs]][link-docs] -->

[badge-tests]: https://img.shields.io/github/actions/workflow/status/imsb-uke/nichepca/test.yaml?branch=main
[link-tests]: https://github.com/imsb-uke/nichepca/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/nichepca

Package for PCA-based spatial domain identification in single-cell spatial transcriptomics data. The corresonding manuscript was published in [Bioinformatics](https://academic.oup.com/bioinformatics/article/41/1/btaf005/7945104?).

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
### Multi-sample support

If you have multiple samples in `adata.obs['sample']`, you can provide the key `sample` to `npc.wf.nichepca` this uses harmony by default:

```python
npc.wf.nichepca(adata, knn=25, sample_key="sample")
```

If you have cell type labels in `adata.obs['cell_type']`, you can directly provide them to `nichepca` as follows (we found this sometimes works better for multi-sample domain identification). However, in this case we need to run `npc.cl.leiden_unique` to handle potential duplicate embeddings:

```python
npc.wf.nichepca(adata, knn=25, obs_key='cell_type', sample_key="sample")
npc.cl.leiden_unique(adata, use_rep="X_npca", resolution=0.5, n_neighbors=15)
```

### Customization

The `nichepca` functiopn also allows to customize the original `("norm", "log1p", "agg", "pca")` pipeline, e.g., without median normalization:
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

## Setting parameters
We found that higher number of neighbors e.g., `knn=25` lead to better results in brain tissue, while `knn=10` works well for kidney data. We recommend to qualitatively optimize these parameters on a small subset of your data. The number of PCs (`n_comps=30` by default) seems to have negligible effect on the results.
## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) or [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).

To create a new conda environment with Python 3.10:

```bash
conda create -n npc-env python=3.10 -y
conda activate npc-env
```

Install the latest development version:

```bash
pip install git+https://github.com/imsb-uke/nichepca.git@main
```

## Contributing

If you want to contribute you can follow this [guide](https://scanpy.readthedocs.io/en/latest/dev/index.html). In short fork the repository, setup a dev environment using this command:

```bash
conda create -n npc-dev python=3.10 -y
conda activate npc-dev
git clone https://github.com/{your-username}/nichepca.git
pip install -e ".[dev, test]"
```
And then make your changes, run the tests and submit a pull request.

## Release notes

See the [changelog][changelog].

## Contact

For questions, help requests, and bug reports, please use the [issue tracker][issue-tracker].

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

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/imsb-uke/nichepca/issues
[changelog]: https://nichepca.readthedocs.io/latest/changelog.html
[link-docs]: https://nichepca.readthedocs.io
[link-api]: https://nichepca.readthedocs.io/latest/api.html
[link-pypi]: https://pypi.org/project/nichepca
