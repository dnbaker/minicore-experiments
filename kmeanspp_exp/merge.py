#!/usr/bin/env python3

"""
Merge script for minicore kmeans++ results
"""

import re
import sys

fns = ['kmpp.dense.p0.1.4threads.out',
       'kmpp.dense.prior1.16threads.out']

# https://github.com/dnbaker/minicore/blob/dev/docs/msr.md
# (though as of 3/8/2021, that's missing e.g. entry 11)
distance_map = {'0': 'L1',
                '1': 'L2',
                '2': 'SQRL2',
                '3': 'JSM',
                '4': 'JSD',
                '5': 'MKL',
                '6': 'HELL',
                '7': 'BATMET',
                '8': 'BATDIST',
                '9': 'TVD',
                '10': 'LLR',
                '11': 'RMKL',
                '12': 'UWLLR',
                '13': 'IS',
                '14': 'RIS',
                '23': 'SIS'}

re_num = re.compile('[01-9]+')
re_nonnum = re.compile('[^01-9]+')

def parse_experiment_name(nm):
	"""
	E.g. 'MC_KM++_MSR4_Cost', 'MC_KMLS++_MSR_Time8'
	"""
	orig_name = nm[:]
	toks = nm.split('_')
	if toks[-1].endswith('++Cost'):
		toks[-1] = toks[-1][:-4]
		toks.append('Cost')
	if toks[-1].endswith('++Time'):
		toks[-1] = toks[-1][:-4]
		toks.append('Time')
	assert 'time' in toks[-1].lower() or 'cost' in toks[-1].lower(), toks
	is_cost = 'cost' in toks[-1].lower()
	name = '_'.join(toks[0:-1])
	dist = None
	if name.startswith('SKL'):
		dist = 'SQRL2'
	else:
		name_num = re_nonnum.sub('', name)
		last_num = re_nonnum.sub('', toks[-1])
		if len(name_num) == 0 and len(last_num) == 0:
			dist = 'SQRL2'
		else:
			num = name_num if len(name_num) > 0 else last_num
			dist = distance_map[num]
	name = re_num.sub('', name)
	assert len(name) > 0
	if name.endswith('_MSR'):
		name = name[:-4]
	is_sparse = True
	if 'Dense' in name:
		name = name.replace('Dense', '')
		is_sparse = False
	name = name.replace('plusplus', 'pp')
	name = name.replace('++', 'pp')
	name = name.replace('_and_', '')
	name = name.replace('KMppLS', 'KMLS')
	name = name.replace('SKL_KMpp', 'SKL')
	print((orig_name, name, dist, is_sparse, is_cost), file=sys.stderr)
	return (name, dist, is_sparse, is_cost, orig_name)


def is_integer(n):
	try:
		float(n)
	except ValueError:
		return False
	else:
		return float(n).is_integer()


def is_float(n):
	try:
		float(n)
	except ValueError:
		return False
	return True


print(','.join(['dataset', 'orig_name', 'k', 'nthreads', 'name', 'distance', 'sparsity', 'measure', 'value']))
for fn in fns:
	with open(fn, 'rt') as fh:
		name_col = 0
		k_col = 1
		nthreads_col = None
		results = {}
		experiments = {}
		for ln in fh:
			if ln.startswith('#Name'):
				toks = ln.rstrip().split()
				assert toks[0] == '#Name', toks[0]
				assert toks[1] == 'k', toks[1]
				for i, tok in enumerate(toks[2:]):
					coli = i+2
					if tok == 'nthreads':
						nthreads_col = coli
					else:
						expt_name, dist, is_sparse, is_cost, orig_name = parse_experiment_name(tok)
						assert coli not in experiments
						experiments[coli] = (expt_name, dist, is_sparse, is_cost, orig_name)
				assert nthreads_col is not None
			elif ln[0] == '#':
				pass
			else:
				assert len(experiments) > 0
				toks = ln.rstrip().split()
				assert is_integer(toks[nthreads_col]), (toks[nthreads_col], toks, nthreads_col)
				name, k, nthreads = toks[name_col], int(toks[k_col]), int(toks[nthreads_col])
				for coli, tup in experiments.items():
					assert is_float(toks[coli])
					expt_name, dist, is_sparse, is_cost, orig_name = tup
					print(','.join([name, orig_name, str(k), str(nthreads)]), end=',')
					print(','.join([expt_name, dist]), end=',')
					print(','.join(['sparse' if is_sparse else 'dense']), end=',')
					print(','.join(['cost' if is_cost else 'time', toks[coli]]))
