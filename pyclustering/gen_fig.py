import numpy as np
import sklearn
import minicore as mc
from time import time as TT
import minicore as mc
import sys
from sklearn.datasets import make_blobs
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
mc.set_num_threads(1)

import argparse
a = argparse.ArgumentParser()
a.add_argument("-o", "--output", default="/dev/stdout")
a.add_argument("-F", "--floating", action='store_true')
a.add_argument("--add-sample-n", action='append')

args = a.parse_args()
samplens = list(map(int, args.add_sample_n)) if args.add_sample_n else []
of = open(args.output, "w")

ns = []
nf = []
k = []
mctimes = []
pctimes = []
mccosts, pccosts = [], []
print("#NumberSamples\tNumFeat\tK\tMCTime_s\tPCTime_s\tMCCost\tPCCost", file=of)
for nsamples in [50000, 100, 10000] + samplens:
    for nfeat in [5, 50, 500, 5000]:
        for K in [5, 25, 50]:
            print(f"{nsamples}, {nfeat}, {K}", file=sys.stderr)
            data, labels = make_blobs(nsamples, nfeat)
            if args.floating: data = data.astype(np.float32)
            s = TT()
            o = mc.kmeanspp(data, k=K, msr=2)
            st = TT()
            pco = kmeans_plusplus_initializer(data, K).initialize()
            st3 = TT()
            ns.append(nsamples)
            nf.append(nfeat)
            mctimes.append(st - s)
            pctimes.append(st3 - st)
            mccosts.append(np.sum(o[2]))
            pccosts.append(np.sum(np.square(np.min(np.array([[np.linalg.norm(pco[i] - x) for x in data] for i in range(len(pco))]), axis=0))))
            print(f"Ratio for {nsamples},{nfeat},{K}: %g" % (pctimes[-1] / mctimes[-1]), file=sys.stderr)
            print("\t".join(map(str, (nsamples, nfeat, K, mctimes[-1], pctimes[-1], mccosts[-1], pccosts[-1]))), file=of)
