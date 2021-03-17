import numpy as np
def loadcao4m():
    import minicore as mc
    pref = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/4MEXP"
    dat = np.fromfile(pref + "/cao4m.data.u16.npy", dtype=np.uint16)
    idx = np.fromfile(pref + "/cao4m.indices.u16.npy", dtype=np.uint16)
    ip = np.fromfile(pref + "/cao4m.indptr.u32.npy", dtype=np.uint32)
    shape = np.fromfile(pref + "/cao4m.shape.u32.npy", dtype=np.uint32)
    return mc.csr_tuple(data=dat, indices=idx, indptr=ip, shape=shape, nnz=len(dat))

def getmat(name):
    if name == "zeisel":
        from load_dropviz import zeisel_cns_mat as ret
    elif name == 'cao':
        from load_cao import cao_mat as ret
    elif name == 'pbmc':
        from load_pbmc import pbmc_mat as ret
    elif name == '293t':
        from load_293t import t293_mat as ret
    elif name == "1.3M":
        from load_1M import million_mat as ret
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
    "cao2d_klnmf": lambda: getmat("cao2dkl")
}

labels = {"cao4m": lambda: np.fromfile("/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/4MEXP/labels.u8.npy", dtype=np.uint8),
          "cao2m": lambda: np.fromfile("/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/minicore-experiments/cao2m.labels.int8.npy", dtype=np.uint8),
          "pbmc": lambda: np.array(list(map(int, map(str.strip, open("/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/minicore-experiments/68k_labels.txt"))))) - 1}

exp_loads['cao2m'] = exp_loads['cao']

ordering = ['293t', 'pbmc', 'zeisel', 'cao', '1.3M', 'cao4m']

__all__ = ["ordering", "exp_loads", "labels"]
