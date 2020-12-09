import numpy as np, scipy.sparse as sp, minicore as mc
from load_lazy import exp_loads
import argparse as ap
from time import time as gett
import multiprocessing
import sys
import random
import uuid
np.random.seed(13)

ps = ap.ArgumentParser("Kmeans++ Experiment")

ps.add_argument("--msr", '-m', type=int, default=5)
ps.add_argument("--prior", '-P', type=float, default=0.)
ps.add_argument("--dataset", type=str, choices=["pbmc", "cao", "zeisel", "293t", "1.3M", "cao4m"], default="pbmc")
ps.add_argument("--maxiter", '-M', type=int, default=5)
ps.add_argument("--ncheckins", '-C', type=int, default=1)
ps.add_argument("--seed", type=int, default=0)
ps.add_argument("paths", nargs='+')
ps.add_argument("--mbsize", "-B", action='append')
ps.add_argument("--coreset-size", "-S", action='append')
ps.add_argument("--ncoresets", "-N", type=int, default=1)

args = ps.parse_args()
ncoresets = args.ncoresets;

mbsizes = args.mbsize if args.mbsize else [100, 500, 2500, 5000, -1]
mbsizes = np.array(mbsizes, dtype=np.uint32)

coresetsizes = args.coreset_size if args.coreset_size else [100, 500, 2500, 5000, 50000]
coresetsizes = np.array(coresetsizes, dtype=np.uint32)

#mc.set_num_threads(args.nthreads)


def load_ctrs(path):
    toks = path.split(":")
    try:
        ds, k, nthreads, nkmc, msr, prior = toks[:6]
        prior = float(prior[5:].split(".u32")[0])
    except:
        print(toks)
        ds, k, nthreads, nkmc, msr = toks[:5]
        msr = msr.split(".")[0]
        prior = 1. if int(msr) in {5, 11, 12, 13, 14, 16, 30, 31} else 0.
    ds = ds.split("/")[-1]
    nthreads = int(nthreads)
    nkmc = int(nkmc)
    k = int(k)
    assert ds in exp_loads.keys()
    return {"dataset": ds, 'k': k, 'p': nthreads,
            'msr': int(msr), 'prior': prior, 'centers': np.fromfile(path, dtype=np.uint32)}

def run_opt(mat, ctrs, mbsizes, *, prior, msr, dataset, nthreads):
    k = ctrs.shape[0]
    ndim = ctrs.shape[1]
    for mbsize in mbsizes:
        print("Processing %d in %s" % (mbsize, str(mbsizes)), file=sys.stderr)
        start = gett()
        out = mc.cluster(mat, ctrs, prior=prior, msr=msr, mbsize=mbsize, maxiter=50)
        stop = gett()
        print(f"{dataset}\t{k}\t{prior}\t{msr}\t{mbsize}\t{nthreads}\t{stop - start}\tout['initcost']\t{out['finalcost']}\t{out['numiter']}\t-1\t-1\n")
        ctrs = out['centers']
        lens = [len(x[0]) for x in ctrs]
        densectrs = np.vstack([sp.csr_matrix((ctrs[i][0], ctrs[i][1], [0, lens[i]]), shape=(1, ndim)).todense() for i in range(len(lens))])
        densectrs.astype(np.float32).tofile(f"ctrs.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.result.f32.{densectrs.shape[0]}.{densectrs.shape[1]}.npy")
        out['asn'].astype(np.uint32).tofile(f"asn.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.result.u32.npy")
        out['costs'].astype(np.float32).tofile(f"costs.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.result.f32.npy")

    print("Getting coreset from initial centers and limited EM", file=sys.stderr)
    initials = mc.cluster(mat, ctrs, prior=prior, msr=msr, maxiter=1)
    costs = initials['costs']
    sampler = mc.CoresetSampler()
    from mc.constants import SENSDICT
    if msr in [0, 1]:
        sens = SENSDICT["VX"]
    elif msr in {3, 5, 7, 9, 11, 12, 13, 14, 16, 17, 30, 31}:
        sens = SENSDICT["LBK"]
    else:
        sens = SENSDICT["BFL"]
    sampler.make_sampler(k, costs, initials['asn'], seed=np.random.choice(range(1000000)))
    for coreset_size in coresetsizes:
        for i in range(ncoresets):
            start = gett()
            weights, items = sampler.sample(coreset_size)
            weights = weights.astype(np.float64)
            (a, b, c), d = mat.rowsel(items)
            ctrs = sp.csr_matrix((a, b, c), shape=d).todense()
            csout = mc.cluster(mat, ctrs, prior=prior, msr=msr, weights=weights)
            stop = gett()
            octrs = csout['centers']
            print(f"{dataset}\t{k}\t{prior}\t{msr}\t{mbsize}\t{nthreads}\t{stop - start}\tcsout['initcost']\t{csout['finalcost']}\t{csout['numiter']}\t{coreset_size}\t{i}\n")
            lens = list(map(len, octrs))
            densectrs = np.vstack([sp.csr_matrix((octrs[i][0], octrs[i][1], [0, lens[i]]), shape=(1, ndim)).todense() for i in range(len(lens))])
            densectrs.astype(np.float32).tofile(f"ctrs.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.result.f32.{densectrs.shape[0]}.{densectrs.shape[1]}.npy")
            csout['asn'].astype(np.uint32).tofile(f"asn.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.{coresetsize}.id{i}.result.u32.npy")
            csout['costs'].astype(np.float32).tofile(f"costs.{dataset}.{msr}.{prior}.{mbsize}.{nthreads}.{coresetsize}.id{i}.result.f32.npy")


print("dataset\tk\tprior\tmsr\tmbsize\tnthreads\truntime\tinitcost\tfinalcost\tnumiter\tcoreset_size\ttrial #\n")

for path in args.paths:
    inf = load_ctrs(path)
    nt = max(inf['p'], 1)
    mc.set_num_threads(nt)
    mat = exp_loads[inf['dataset']]()
    if (mat.data.dtype.kind == 'i' and mat.data.dtype.itemsize > 2) or mat.data.dtype is np.dtype("float64"):
        mat.data = mat.data.astype(np.float32)
    sm = mc.CSparseMatrix(mat)
    sel = sm.rowsel(inf['centers'])
    ctrs = sp.csr_matrix(tuple(sel[:3]), shape=sel[-1]).astype(np.float64).todense()
    run_opt(sm, ctrs, mbsizes, prior=args.prior, msr=inf['msr'], dataset=inf['dataset'], nthreads=nt)
