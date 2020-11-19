import minicore as mc, scipy.sparse as sp, numpy as np
from collections import Counter

ns = 50
asn = np.random.choice(range(5), size=(ns,))
costs = np.abs(np.random.standard_cauchy(size=(ns,))).astype(np.float32)
costs[costs == 0.] = 1.;
print(sorted(enumerate(costs), key=lambda x: -x[1])[:20])
print(Counter(asn))
for name, ival in mc.constants.SENSDICT.items():
    print(name, ival)
    if name == "LFKF": continue
    cs = mc.CoresetSampler()
    cs.make_sampler(5, costs, assignments=asn, sens=ival)
    samps = cs.sample(10)
