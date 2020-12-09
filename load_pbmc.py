import numpy as np
from scipy.io import mmread
import scipy.sparse as sp
import pickle
import minicore as mc

'''Loads the PBMC experiment dataset,
   ensures that the types are correct and the sizes match
'''

PATH="/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/clusterdash/minocore/scripts/pbmc.bin.pkl"

pbmc_mat = pickle.load(open(PATH, "rb"))

pbmc_mat = mc.csr_tuple(data=pbmc_mat.data.astype(np.uint16), indices=pbmc_mat.indices.astype(np.uint16), indptr=pbmc_mat.indptr, shape=pbmc_mat.shape, nnz=len(pbmc_mat.data))

__all__ = ["pbmc_mat"]
