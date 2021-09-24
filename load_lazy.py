import numpy as np
import scipy.sparse as sp
import os
import lzma
import sys


PREFIX=os.environ.get("MINICORE_DATA", "")

if not PREFIX:
    print("MINICORE_DATA must be set for data loading; data available at https://zenodo.org/record/4738365. Download, decompress, and set MINICORE_DATA to the path.", file=sys.stderr)
    raise ImportError()


def loadcao4m():
    pref = f"{PREFIX}/CAO/4m/cao4m"
    import minicore as mc
    dat = np.fromfile(f"{pref}.data.u16.npy", dtype=np.uint16)
    idx = np.fromfile(f"{pref}.indices.u16.npy", dtype=np.uint16)
    ip = np.fromfile(f"{pref}.indptr.u32.npy", dtype=np.uint32)
    shape = np.fromfile(f"{pref}.shape.u32.npy", dtype=np.uint32)
    return mc.csr_tuple(data=dat, indices=idx, indptr=ip, shape=shape, nnz=len(dat))


def load293t():
    pref = f"{PREFIX}/293T"
    dat = np.fromfile(f"{pref}/data.u16.npy", dtype=np.uint16)
    idx = np.fromfile(f"{pref}/indices.u16.npy", dtype=np.uint16)
    ip = np.fromfile(f"{pref}/indptr.u32.npy", dtype=np.uint32)
    shape = np.fromfile(f"{pref}/shape.u32.npy", dtype=np.uint32)
    return sp.csr_matrix((dat, idx, ip), shape=shape)


def getmat(name):
    if name == "zeisel":
        pref = f"{PREFIX}/ZEISEL/"
        return sp.csr_matrix((np.fromfile(f"{pref}data.u16.npy", dtype=np.uint16),
            np.fromfile(f"{pref}indices.u16.npy", dtype=np.uint16),
            np.fromfile(f"{pref}indptr.u32.npy", dtype=np.uint32)
        ), shape=np.fromfile(f"{pref}shape.u32.npy", dtype=np.uint32))
    elif name.lower() in ['cao', 'cao2m', 'cao2']:
        pref = f"{PREFIX}/CAO/2m/cao2m."
        return sp.csr_matrix((np.frombuffer(lzma.open(f"{pref}data.u16.npy.xz", "rb").read(), dtype=np.uint16),
            np.fromfile(f"{pref}indices.u16.npy", dtype=np.uint16),
            np.frombuffer(lzma.open(f"{pref}indptr.i64.npy.xz", "rb").read(), dtype=np.int64)
        ), shape=np.fromfile(f"{pref}shape.u32.npy", dtype=np.uint32))
    elif name == 'pbmc':
        pref = f"{PREFIX}/PBMC/pbmc."
        return sp.csr_matrix((np.frombuffer(lzma.open(f"{pref}data.u16.npy.xz", "rb").read(), dtype=np.uint16),
            np.frombuffer(lzma.open(f"{pref}indices.u16.npy.xz", "rb").read(), dtype=np.uint16),
            np.frombuffer(lzma.open(f"{pref}indptr.u32.npy.xz", "rb").read(), dtype=np.uint32)
        ), shape=np.fromfile(f"{pref}shape.u32.npy", dtype=np.uint32))
    elif name == '293t':
        return load293t()
    elif name == "1.3M":
        pref = f"{PREFIX}/1M/1M."
        return sp.csr_matrix((np.fromfile(f"{pref}data.u16.npy", dtype=np.uint16),
            np.fromfile(f"{pref}indices.u16.npy", dtype=np.uint16),
            np.fromfile(f"{pref}indptr.i64.npy", dtype=np.uint64)
        ), shape=np.fromfile(f"{pref}shape.u32.npy", dtype=np.uint32))
    elif name == "cao2d":
        return np.fromfile(f"{PREFIX}/Dense/cao2m.pca.2058652.500.float32", dtype=np.float32).reshape((2058652, 500))
    elif name == "pbmcd":
        return np.fromfile(f"{PREFIX}/Dense/pbmc_pca500.f32.68579.500.npy", dtype=np.float32).reshape((68579, 500))
    elif name == 'cao4d':
        return np.fromfile(f"{PREFIX}/Dense/cao4m.pca.4062980.500.float32", dtype=np.float32).reshape((4062980, 500))
    elif name.lower() in ['cao2dkl', 'cao2dklnmf']:
        return np.fromfile(f"{PREFIX}/Dense/cao2m.2058652.500.kl.nmf.float32", dtype=np.float32).reshape((2058652, 500))
    elif name == 'cao4m':
        ret = loadcao4m()
    elif name == 'pbmcd':
        import load_dense
        return load_dense.pbmcd
    elif name == 'cao4d':
        import load_dense
        return load_dense.cao4d
    elif name == 'cao2d':
        import load_dense
        return load_dense.cao2d
    elif name == 'cao2dkl':
        import load_dense
        return load_dense.cao2d_klnmf
    else:
        raise RuntimeError("Not found: name")
    return ret

exp_loads = {
    "cao": lambda: getmat("cao"),
    "zeisel": lambda: getmat("zeisel"),
    "293t": lambda: getmat("293t"),
    "pbmc": lambda: getmat("pbmc"),
    "1.3M": lambda: getmat("1.3M"),
    "cao4m": lambda: getmat('cao4m'),
    "cao2d": lambda: getmat("cao2d"),
    "cao4d": lambda: getmat("cao4d"),
    "pbmcd": lambda: getmat("pbmcd"),
    "cao2d_klnmf": lambda: getmat("cao2dkl"),
    "cao2dkl": lambda: getmat("cao2dkl")
}

labels = {
          "cao4m": lambda: np.fromfile(f"{PREFIX}/CAO/4m/labels.u8.npy", dtype=np.uint8),
          "cao2m": lambda: np.fromfile(f"{PREFIX}/CAO/2m/labels.u8.npy", dtype=np.uint8),
          "pbmc": lambda: np.fromfile(f"{PREFIX}/PBMC/labels.u8.npy", dtype=np.uint8)
}

exp_loads['cao2m'] = exp_loads['cao']

ordering = ['293t', 'pbmc', 'zeisel', 'cao', '1.3M', 'cao4m']

__all__ = ["ordering", "exp_loads", "labels"]
