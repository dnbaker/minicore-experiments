import sys

import numpy as np
import scipy.sparse as sp
import sklearn.cluster as skc
import minicore as mc

from time import time
from multiprocessing import cpu_count

from load_lazy import *

from argparse import ArgumentParser as AP
ap = AP()
ap.add_argument("--nthreads", "-p", default=32, type=int)
ap.add_argument("--tol", "-T", default=1e-5, type=float)
args = ap.parse_args()
#pbmc = [exp_loads[x]() for x in ['pbmc']][0]
pbmc, cao2, cao4 = [exp_loads[x]() for x in ['pbmc', 'cao', 'cao4m']]

t0 = time()
pbmc_csr = sp.csr_matrix((pbmc.data, pbmc.indices, pbmc.indptr), shape=pbmc.shape)
pbmc_csm = mc.CSparseMatrix(pbmc_csr)
print(f"pmbc load: {time() - t0}", file=sys.stderr)

t0 = time()
cao4_csr = sp.csr_matrix((cao4.data, cao4.indices, cao4.indptr), shape=cao4.shape)
cao4_csm = mc.CSparseMatrix(cao4)
print(f"cao4 load: {time() - t0}", file=sys.stderr)

t0 = time()
cao2_csr = cao2
cao2_csm = mc.CSparseMatrix(cao2)
print(f"cao2 load: {time() - t0}", file=sys.stderr)

mc.set_num_threads(args.nthreads)
print("Loaded data, processing with %d threads" % mc.get_num_threads(), file=sys.stderr)
KSET = [3, 10, 25, 50, 100]
print("#Name\tk\tSKL KM++ time\tSKL KM++ cost\tMC KM++ time\tMC KM++ cost\tMC KM++ plus LS++\tMC KM++ plus LS++ runtime\tSKL KM time\tSKL KM cost\tMC MBKM time\tMC MBKM cost\tMC EMKM time\tMC EMKM cost\tMC CSKM time\tMC CSKM cost")

def print_set(csr, csm, *, name, kset=KSET):
    for k in kset:
        if k > 50 and csr.shape[0] > 1000000: continue
        t = time()
        try:
            ctrs, ids = skc.kmeans_plusplus(csr, n_clusters=k, n_local_trials=1)
        except TypeError as e:
            csr.indices = csr.indices.astype(np.uint32)
            csr.indptr = csr.indptr.astype(np.uint32)
            ctrs, ids = skc.kmeans_plusplus(csr, n_clusters=k, n_local_trials=1)
        t2 = time()
        costs = np.array([[np.linalg.norm(ctr - csr[rid]) for ctr in ctrs] for rid in range(csr.shape[0])])
        skcost = np.sum(np.min(costs, axis=1))
        t3 = time()
        mco = mc.kmeanspp(csm, k=k, ntimes=1, msr="SQRL2")
        t4 = time()
        print(f"{name}\t{k}\t{t2 - t}\t{skcost}\t{t4 - t3}\t{np.sum(mco[2])}", end='\t', flush=True)
        t5 = time()
        mcols = mc.kmeanspp(csm, k=k, ntimes=1, msr="SQRL2", lspp=3 * k)
        t6 = time()
        print(f"{np.sum(mcols[2])}\t{t6 - t5}", end='\t', flush=True)
    
        t = time()
        skkm = skc.KMeans(k, init=ctrs, tol=args.tol, n_init=1)
        skkmcost = 0.
        try:
            skkm.fit(csr)
            skkmcost = skkm.inertia_
        except ValueError as e:
            skkmcost = 1.e308
            print("Failed to fit CSR matrix", file=sys.stderr)
        t2 = time()
            
        
        t7 = time()
        mckmo = mc.hcluster(csm, list(map(int, mco[0])), msr="SQRL2", eps=args.tol, mbsize=min(csr.shape[0], 5000))
        t8 = time()
        t9 = time()
        mckmo_cs = mc.hcluster(csm, list(map(int, mco[0])), msr="SQRL2", eps=args.tol, use_cs=True, mbsize=min(csr.shape[0], 5000))
        t10 = time()
        t3 = time()
        mckmo_em = mc.hcluster(csm, list(map(int, mco[0])), msr="SQRL2", eps=args.tol)
        t4 = time()
        print(f"{t2 - t}\t{skkmcost}\t{t8 - t7}\t{mckmo['finalcost']}\t{t4 - t3}\t{mckmo_em['finalcost']}\t{t10 - t9}\t{mckmo_cs['finalcost']}", flush=True)

print_set(pbmc_csr, pbmc_csm, name="pbmc")
print_set(cao4_csr, cao4_csm, name="Cao4m")
print_set(cao2_csr, cao2_csm, name="Cao2m")
