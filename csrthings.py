import numpy as np, scipy.sparse as sp, minicore as mc
from load_lazy import exp_loads
import argparse as ap
from time import time as gett
import sys
import random
ps = ap.ArgumentParser()

ps.add_argument("--msr", '-m', type=int, default=5)
ps.add_argument("--prior", '-P', type=float, default=0.)
ps.add_argument('--nthreads', '-p', type=int, default=1)
ps.add_argument("--dataset", type=str, choices=["pbmc", "cao", "zeisel", "293t", "1.3M"], default="pbmc")
ps.add_argument("--mbsize", '-B', type=int, default=-1)
ps.add_argument("-k", type=int, default=5)
ps.add_argument("--maxiter", '-M', type=int, default=5)
ps.add_argument("--ncheckins", '-C', , type=int, default=1)

BATCH_SIZES = [50, 500, 5000, -1]

args = ps.parse_args()

mc.set_num_threads(args.nthreads)

matcao = exp_loads[args.dataset](); csrcao = sp.csr_matrix(matcao);
csrcao.indices = csrcao.indices.astype(np.uint32)
smcao = mc.CSparseMatrix(csrcao)

st = gett()
caokm = mc.kmeanspp(smcao, seed=13, msr=args.msr, k=args.k, betaprior=args.prior)
et = gett()
print("kmeans++ in %fs" % (et - st), file=sys.stderr)
subc_sp = sp.vstack([csrcao[x] for x in caokm[0]])
subc = mc.smw(subc_sp)

# ev = mc.cmp(smcao, subc, betaprior=args.prior, msr=args.msr)
# print(ev)
st = gett()
cao_em5 = mc.cluster_from_centers(smcao, subc_sp.todense(), msr=args.msr, maxiter=args.maxiter, ncheckins=5, reseed_count=0, betaprior=args.prior, mbsize=args.mbsize, with_rep=False,
                                  seed=random.randint(0, 1<<32))
et = gett()
print("kmeans %s in %fs" % (f"w minibatch={args.mbsize}" if args.mbsize > 0 else "via em",  et - st), file=sys.stderr)
