This repository contains experiments for `minicore: Fast scRNA-seq clustering with various distances`.



### Dependencies
1. sklearn v0.12
2. numpy v0.19
3. scipy v1.5
4. minicore v0.4

### Installing minicore
To install minicore's Python bindings, clone [minicore recursively](https://github.com/dnbaker/minicore), change into the Python directory and run `python3 setup.py install`.

Because compilation can be somewhat slow, the build is parallelized by default with `$OMP_NUM_THREADS` threads. Run your command as `OMP_NUM_THREADS=number python3 setup.py install` to use `number` threads.


### Setting up data

Because these datasets are large, you will need to set up paths to the data on disk in order to run these experiments. The changes will be in the `load_*.py` files.
