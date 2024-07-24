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

Given an AnnData object `adata`, you can run nichepca as follows:

```python
import scanpy as sc
import nichepca as npc

npc.wf.run_nichepca(adata, knn=5)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.5)
```

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install nichepca:

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