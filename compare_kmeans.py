from load_lazy import exp_loads, ordering

from argparse import ArgumentParser as ap

a = ap()
a.add_argument("experiment", help="must be 'cao', 'pbmc', '1.3M', '293t', 'zeisel'")

args = a.parse_args()
