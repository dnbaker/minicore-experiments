import sys

sys.path.append("..")
from time import time
from sklearn.datasets import make_classification

import numpy as np
import scipy.sparse as sp
import sklearn.cluster as skc
import minicore as mc

from load_lazy import *

from argparse import ArgumentParser as AP
ap = AP()
ap.add_argument("--tol", "-T", default=1e-5, type=float)
ap.add_argument("--maxiter", default=25, type=int)
ap.add_argument("--ncheckins", default=5, type=int)
ap.add_argument("--n-local-trials", type=int, default=1)
ap.add_argument("--mbsize", type=int, default=5000)
ap.add_argument("--msr", '-M', action='append')
ap.add_argument("--prior", type=float, default=1.)
args = ap.parse_args()
maxiter=args.maxiter
nlt = args.n_local_trials

print("#args=" + str(args))

print(f"#Name\tSKLSparse_KMplusplus_time\tSKLSparse_KMplusplus_cost\tSKLDense_KMplusplus_time\tSKLDENSE_KMplusplus_cost\tnthreads_always_1", end="")

fakedata = make_classification(n_samples=50000, n_classes=10, n_clusters_per_class=4, n_informative=50, n_features=100)[0]
fakedata[np.abs(fakedata) < 1.] = 0.
fakedata = np.abs(fakedata)
fdsums = np.sum(fakedata, axis=1)
for i, s in enumerate(fdsums):
    if not s:
        fakedata[i,np.random.choice(1000, size=3, replace=False)] = 1.

nzc = np.sum(fakedata != 0, axis=1)
# print("Nnza: %g, %g\n" % (np.mean(fdsums), np.median(fdsums)))
tiny_csr = sp.csr_matrix(fakedata.astype(np.float32))
tiny_csm = mc.CSparseMatrix(tiny_csr)
KSET = [3, 10, 25, 50, 100]

msrs = []
for m in args.msr if args.msr else []:
    try:
        msrs.append(int(m))
    except:
        msrs.append(m)
print("\t" + "\t".join(f"MC_KM++_MSR_Time{m}\tMC_KM++_MSR{m}_Cost\tMC_KMLS++_MSR_Time{m}\tMC_KMLS++_MSR{m}_Cost" for m in msrs))

def k2lsppn(k):
    return int(np.ceil(k + 3 + np.log(k)))

def print_set(csr, csm, dmat, *, name, kset=KSET):
    if callable(dmat):
        dmat = dmat()
    smw = mc.smw(csr.astype(np.float32))
    for k in kset:
        lsppn = k2lsppn(k)
        print(f"Performing k={k} for name = {name}", file=sys.stderr)
        t = time()
        try:
            ctrs, ids = skc.kmeans_plusplus(csr, n_clusters=k, n_local_trials=nlt)
        except (TypeError, ValueError) as e:
            csr.indices = csr.indices.astype(np.uint32)
            csr.indptr = csr.indptr.astype(np.uint32)
            ctrs, ids = skc.kmeans_plusplus(csr, n_clusters=k, n_local_trials=nlt)
        try:
            skcost = np.sum(np.min(mc.cmp(smw, csr[np.array(sorted(ids))].todense()), axis=1))
        except:
            print("Exception thrown, skcost is infinite", file=sys.stderr)
            skcost = 1e308
        t2 = time()
        print(f"{name}\t{k}\t{t2 - t}\t{skcost}", end='\t', flush=True)
        t3 = time()
        print(dmat,type(dmat),file=sys.stderr)
        dctrs, dids = skc.kmeans_plusplus(dmat, n_clusters=k, n_local_trials=nlt)
        t4 = time()
        skcost = np.sum(np.min(mc.cmp(smw, csr[np.array(sorted(ids))].todense()), axis=1))
        print(f"{t4 - t3}\t{skcost}\t1", flush=True, end='\n')


print_set(tiny_csr, tiny_csm, tiny_csr.todense(), name="tiny")
print("loading data from disk...\n", file=sys.stderr)
pbmc, cao2, cao4, pbmcd, cao2d, cao4d = [exp_loads[x]() for x in ['pbmc', 'cao', 'cao4m', 'pbmcd', 'cao2d', 'cao4d']]

print("loaded data from disk...\n", file=sys.stderr)

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

print("Loaded data, processing with %d threads" % mc.get_num_threads(), file=sys.stderr, flush=True)
print_set(pbmc_csr, pbmc_csm, pbmcd, name="PBMC")
print_set(cao4_csr, cao4_csm, cao4d, name="Cao4m")
print_set(cao2_csr, cao2_csm, cao2d, name="Cao2m")
