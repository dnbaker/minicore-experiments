import sys
import minicore as mc
import numpy as np
from argparse import ArgumentParser as AAP
from time import time as tt
ap = AAP()
aa = ap.add_argument
aa("--nthreads", '-p', type=int, help="Number of threads")
aa("--memmap", action='store_true', help="Whether or not to memory-map the sparse vectors")
aa("-k", type=int, default=10)
aa("prefix", help="Prefix for compressed files")
aa("--msr", "-M", default="MKL")
aa("--prior", "-P", default=0.01, type=float)
aa("--mbsize", type=int, default=10000)
aa("--maxiter", type=int, default=10)
aa("--ncheckins", type=int, default=2)

args = ap.parse_args()
prefix = args.prefix

if args.memmap:
    data, indices = map(lambda x: np.memmap(x, dtype=np.uint16), (prefix + "data.u16.npy", prefix + "indices.u16.npy"))
    indptr, shape = map(lambda x: np.memmap(x,  dtype=np.uint32), (prefix + "indptr.u32.npy", prefix + "shape.u32.npy"))
else:
    data, indices = map(lambda x: np.fromfile(x, dtype=np.uint16), (prefix + "data.u16.npy", prefix + "indices.u16.npy"))
    indptr, shape = map(lambda x: np.fromfile(x,  dtype=np.uint32), (prefix + "indptr.u32.npy", prefix + "shape.u32.npy"))

mc.set_num_threads(args.nthreads)
mat = mc.csr_tuple(data=data, indices=indices, indptr=indptr, shape=shape, nnz=len(indptr) - 1)
csm = mc.CSparseMatrix(mat)
t1 = tt()
ctrs, asn, costs = mc.kmeanspp(csm, k=args.k, msr=args.msr, prior=args.prior)
t2 = tt()
print(f"kmeans++ with {args.nthreads} threads, k = {args.k} in {t2 - t2}s", file=sys.stdout)
print("cost: " + str(np.sum(costs)))

t1 = tt()
output = mc.hcluster(csm, ctrs, mbsize=args.mbsize, prior=args.prior, msr=args.msr, maxiter=args.maxiter, ncheckins=args.ncheckins)
t2 = tt()
print(f"mbkmeans with {args.nthreads} threads, k = {args.k} in {t2 - t2}s", file=sys.stdout)
