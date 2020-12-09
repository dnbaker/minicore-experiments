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

sumdict = {}

ncomplete = 0
print("#dataset\tmsr\tk\tnthreads\tnkmc\tmsr\tmedtime\tmeancost\tmincost")
for nthreads in [min(64, multiprocessing.cpu_count()), 32, 16]:
    mc.set_num_threads(nthreads)
    for k in KSET:
        for nkmc in KMC2:
            times = np.zeros(NTIMES)
            costs = np.zeros(NTIMES)
            kms = []
            for nt in range(NTIMES):
                st = gett()
                caokm = mc.kmeanspp(smcao, seed=args.seed + nt, msr=args.msr, k=k, betaprior=args.prior, nkmc=nkmc)
                et = gett()
                times[nt] = et - st
                costs[nt] = np.sum(caokm[2])
                kms.append(caokm)
            med = np.median(times)
            meancost = np.mean(costs)
            bestcost = np.argmin(costs)
            kms[bestcost][0].tofile(f"{args.dataset}:{k}:{nthreads}:{nkmc}:{args.msr}:prior{args.prior}.u32.centers.npy")
            key = f"{args.dataset}:{k}:{nthreads}:{nkmc}:{args.msr}"
            value = {"medtime": med, "meancost": meancost, "mincost": np.min(costs)}
            # sumdict[key] = value
            print(f"{args.dataset}\t{args.msr}\t{k}\t{nthreads}\t{nkmc}\t{med}\t{meancost}\t{value['mincost']}", flush=True)
            ncomplete += 1
            print("Completed %d " % ncomplete, file=sys.stderr)
        print("Completed nkmc for k = %d" % k, file=sys.stderr)
    print("Completed experiment for nthreads = %d", nthreads, file=sys.stderr)
