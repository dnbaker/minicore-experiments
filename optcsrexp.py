import numpy as np, scipy.sparse as sp, minicore as mc
from load_lazy import exp_loads
import argparse as ap
from time import time as gett
import multiprocessing
import sys
import random
import uuid
ps = ap.ArgumentParser("Kmeans++ Experiment")

ps.add_argument("--msr", '-m', type=int, default=5)
ps.add_argument("--prior", '-P', type=float, default=0.)
#ps.add_argument('--nthreads', '-p', type=int, default=1)
ps.add_argument("--dataset", type=str, choices=["pbmc", "cao", "zeisel", "293t", "1.3M", "cao4m"], default="pbmc")
ps.add_argument("--mbsize", '-B', type=int, default=-1)
ps.add_argument("--maxiter", '-M', type=int, default=5)
ps.add_argument("--ncheckins", '-C', type=int, default=1)
ps.add_argument("--seed", type=int, default=0)
ps.add_argument("paths", nargs='+')

args = ps.parse_args()

#mc.set_num_threads(args.nthreads)

KSET = [5, 25, 100]
KMC2 = [0]

NTIMES = 1

matcao = exp_loads[args.dataset]();
#csrcao = sp.csr_matrix(matcao, dtype=np.float32)
#csrcao.indices = csrcao.indices.astype(np.uint32)
#csrcao.indptr = csrcao.indptr.astype(np.uint64)
smcao = mc.CSparseMatrix(matcao)

def load_ctrs(path):
    toks = path.split(":")
    try:
        ds, k, nthreads, nkmc, msr, prior = toks[:6]
        prior = float(prior[5:].split(".u32")[0])
    except:
        ds, k, nthreads, nkmc, msr = toks[:5]
        msr = msr.split(".")[0]
        prior = 1. if int(msr) in {5, 11, 12, 13, 16, 30, 31} else 0.
    ds = ds.split("/")[-1]
    nthreads = int(nthreads)
    nkmc = int(nkmc)
    k = int(k)
    assert ds in exp_loads.keys()
    return {"dataset": exp_loads[ds], 'k': k, 'p': nthreads, 'msr': int(msr), 'prior': prior, 'centers': np.fromfile(path, dtype=np.uint32)}

for path in args.paths:
    inf = load_ctrs(path)
    print(inf)
