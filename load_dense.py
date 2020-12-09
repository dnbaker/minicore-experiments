import numpy as np

# The first three were generated with sklearn.decomposition.TruncatedSVD
# and the second, with sklearn.decomposition.NMF with KL divergence
# and only 10 iterations.
# Modify these paths to point to the relevant data
cao2d = np.fromfile("/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/minicore-experiments/dense/cao2m.pca.2058652.500.float32", dtype=np.float32).reshape((2058652, 500))
cao4d = np.fromfile("/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/minicore-experiments/dense/cao4m.pca.4062980.500.float32", dtype=np.float32).reshape((4062980, 500))
pbmcd = np.fromfile("/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/minicore-experiments/dense/pbmc_pca500.f32.68579.500.npy", dtype=np.float32).reshape((68579, 500))
cao2d_klnmf = np.fromfile("/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/minicore-experiments/dense/cao2m.2058652.500.kl.nmf.float32", dtype=np.float32).reshape((2058652, 500))

__all__ = ["cao2d", "cao4d", "pbmcd", "cao2d_klnmf"]
