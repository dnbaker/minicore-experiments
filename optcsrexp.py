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
ps.add_argument("--dataset", type=str, choices=["pbmc", "cao", "zeisel", "293t", "1.3M", "cao4m"], default="pbmc")
ps.add_argument("--maxiter", '-M', type=int, default=5)
ps.add_argument("--ncheckins", '-C', type=int, default=1)
ps.add_argument("--seed", type=int, default=0)
ps.add_argument("paths", nargs='+')
ps.add_argument("--mbsize", "-B", action='append')

args = ps.parse_args()

mbsizes = args.mbsize if args.mbsize else [100, 500, 2500, 5000, -1]
mbsizes = np.array(mbsizes, dtype=np.uint32)

#mc.set_num_threads(args.nthreads)


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
    return {"dataset": ds, 'k': k, 'p': nthreads,
            'msr': int(msr), 'prior': prior, 'centers': np.fromfile(path, dtype=np.uint32)}

def run_opt(mat, ctrs, mbsizes, *, prior, msr, dataset, nthreads):
    k = ctrs.shape[0]
    res = []
    for mbsize in mbsizes:
        start = gett()
        out = mc.cluster(mat, ctrs, prior=prior, msr=msr)
        stop = gett()
        print(f"{dataset}\t{k}\t{prior}\t{msr}\t{mbsize}\t{nthreads}\t{stop - start}\tout['initcost']\t{out['finalcost']}\t{out['numiter']}\n")
        out['centers'].astype(np.uint32).tofile("ctrs.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.result.u32.npy")
        out['asn'].astype(np.uint32).tofile("asn.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.result.u32.npy")
        out['cost'].astype(np.float32).tofile("cost.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.result.f32.npy")
    
print("dataset\tk\tprior\tmsr\tmbsize\tnthreads\truntime\tinitcost\tfinalcost\tnumiter\n")

for path in args.paths:
    inf = load_ctrs(path)
    print(inf)
    nt = max(inf['p'], 1)
    mc.set_num_threads(nt)
    mat = exp_loads[inf['dataset']]()
    sm = mc.CSparseMatrix(mat)
    print("Got matrix, getting centers")
    sel = sm.rowsel(inf['centers'])
    ctrs = sp.csr_matrix(tuple(sel[:3]), shape=sel[-1]).astype(np.float64).todense()
    print(ctrs)
    run_opt(sm, ctrs, mbsizes, prior=args.prior, msr=inf['msr'], dataset=inf['dataset'], nthreads=nt)
