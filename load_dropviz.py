import numpy as np
import scipy.sparse as sp
import minicore as mc

'''Loads the DropViz experiment dataset,
   ensures that the types are correct and the sizes match
'''

PREFIX = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/berger_data/mouse_brain/zeisel/zeisel."

def load_files(prefix):
    data = np.fromfile(prefix + "data.file", dtype=np.uint8).view(np.uint64).astype(np.uint32)
    indices = np.fromfile(prefix + "indices.file", dtype=np.uint8).view(np.uint32)
    indptr = np.fromfile(prefix + "indptr.file", dtype=np.uint8).view(np.uint32)
    shape = np.fromfile(prefix + "shape.file", dtype=np.uint8).view(np.uint32)
    return (data, indices, indptr, shape)


data, indices, indptr, shape = load_files(PREFIX)
zeisel_cns_mat = sp.csr_matrix((data, indices, indptr), shape)
zeisel_cns_mat.indices = zeisel_cns_mat.indices.astype(np.uint16)
zeisel_cns_mat.data = zeisel_cns_mat.data.astype(np.uint16)
zeisel_cns_mat.indptr = zeisel_cns_mat.indptr.astype(np.uint64)

__all__ = ["zeisel_cns_mat"]
