import numpy as np
import scanpy as sc
import scipy
from sklearn.cluster import KMeans


def generate_dummy_adata(n_cells=100, n_genes=50, n_samples=2, n_celltypes=5, seed=0):
    random_state = np.random.RandomState(seed)

    X = random_state.randint(0, 400, size=(n_cells, n_genes))
    mask = random_state.rand(*X.shape) < 0.5
    X[mask] = 0
    X = scipy.sparse.csr_matrix(X)

    adata = sc.AnnData(X)

    # add random spatial coords in the unit square
    coords = random_state.rand(n_cells, 2)
    adata.obsm["spatial"] = coords

    # Partition the spatial coordinates into approximately equal parts
    kmeans = KMeans(n_clusters=n_samples, random_state=seed, n_init="auto")
    samples = kmeans.fit_predict(coords)
    adata.obs["sample"] = [str(s) for s in samples]

    # create artificial cell type column
    adata.obs["cell_type"] = random_state.randint(0, n_celltypes, size=n_cells)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str).astype("category")

    return adata
