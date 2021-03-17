import sys
sys.path.append("..")
from time import time, perf_counter
from sklearn.datasets import make_classification
import numpy as np
import scipy.sparse as sp
import sklearn.cluster as skc
import minicore as mc
from sklearn.metrics import adjusted_rand_score as ARI

from load_lazy import exp_loads, labels

from argparse import ArgumentParser as AP
ap = AP()
ap.add_argument("--nthreads", "-p", default=1, type=int)
ap.add_argument("--maxiter", default=25, type=int)
ap.add_argument("--ncheckins", default=5, type=int)
ap.add_argument("--lspp", type=int, default=0, help="Number of localsearch++ rounds to perform")
ap.add_argument("--n-local-trials", type=int, default=1)
ap.add_argument("--mbsize", type=int, default=5000)
ap.add_argument("--msr", '-M', action='append')
ap.add_argument("--prior", type=float, default=1.)
ap.add_argument("--dataset", type=str, choices=["cao4m", "cao2m", "pbmc"], default="pbmc")
ap.add_argument("--hvg", type=int, default=-1, help="Number of highly variable genes to filter down to. If -1, use the full data.")
ap.add_argument("--densify", action='store_true', help="Densify the matrix after filtering to hvg. Requires hvg")
ap.add_argument("-k", default=25, type=int, help="k for clustering")
args = ap.parse_args()
maxiter=args.maxiter
mc.set_num_threads(args.nthreads)
nlt = args.n_local_trials
if args.densify and args.hvg < 0:
    raise RuntimeError("args.densify requires hvg")

print("#args=" + str(args))

pref = str(perf_counter())[-8:] + "."
print(f"#pref:{pref}")

measures = sorted(set(list(map(int, args.msr)) + [2]))

dataset = exp_loads[args.dataset]()
truelabels = labels[args.dataset]()
if args.hvg > 0:
    t = time()
    hvdata, _, __ = mc.hvg(dataset, args.hvg)
    print(f"#{args.dataset}\tHVGene{args.hvg}\t{time() - t}")
    if args.densify and isinstance(hvdata, sp.csr_matrix):
        t = time()
        hvdata = hvdata.todense()
        print(f"#{args.dataset}\tDensifyHVG{args.hvg}\t{time() - t}")
    dataset = hvdata
print("#Dataset\tk\tnumberHVG\tMeasure\tPrior\tLocalsearchRounds\tNumLocalTrials\tDensified\tMinibatchSize\tKmeansplusplusTime\tKmeansplusplusCost\tKMeansplusplusARI\tKMeansTime\tKMeansCost\tKMeansARI")
for msr in measures:
    t = time()
    if isinstance(dataset, sp.csr_matrix) or isinstance(dataset, mc.csr_tuple):
        init = mc.kmeanspp(mc.CSparseMatrix(dataset), k=args.k, msr=msr, lspp=args.lspp, n_local_trials=args.n_local_trials, prior=args.prior)
    else:
        init = mc.kmeanspp(dataset, msr=msr, k=args.k, lspp=args.lspp, n_local_trials=args.n_local_trials, prior=args.prior)
    td = time() - t
    initcost = np.sum(init[2])
    kmpp_ari = ARI(init[1].astype(np.int32), truelabels)
    print(f"hvg: {args.hvg}\n", file=sys.stderr)
    print(f"{args.dataset}\t{args.k}\t{args.hvg}\t{mc.meas2str(msr)}\t{args.prior}\t{args.lspp}\t{args.n_local_trials}\t{args.densify}\t{args.mbsize}\t{td}\t{initcost}\t{kmpp_ari}", end="")
    t = time()
    if isinstance(dataset, sp.csr_matrix) or isinstance(dataset, mc.csr_tuple):
        cout = mc.hcluster(mc.CSparseMatrix(dataset), init[0], msr=msr, maxiter=args.maxiter, ncheckins=args.ncheckins, prior=args.prior, mbsize=args.mbsize)
    else:
        cout = mc.hcluster(dataset, init[0], msr=msr, maxiter=args.maxiter, ncheckins=args.ncheckins, prior=args.prior, mbsize=args.mbsize)
    td = time() - t
    kmeans_ari = ARI(cout['asn'], truelabels)
    print(f"\t{td}\t{cout['finalcost']}\t{kmeans_ari}")
