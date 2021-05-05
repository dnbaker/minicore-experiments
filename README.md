This repository contains experiments for `minicore: Fast scRNA-seq clustering with various distances`.



### Dependencies
1. sklearn
2. numpy
3. scipy
4. minicore v0.3

### Installing minicore
To install minicore's Python bindings, clone [minicore recursively](https://github.com/dnbaker/minicore), change into the Python directory and run `python3 setup.py install`.

Because compilation can be somewhat slow, the build is parallelized by default with `$OMP_NUM_THREADS` threads. Run your command as `OMP_NUM_THREADS=number python3 setup.py install` to use `number` threads.


### Setting up data

A zenodo will be provided soon with a tarball of the datasets used in the paper. Decompress this file, and then run `export MINICORE_DATA=${PATH}`. This should enable `load_lazy.py` to load all these datasets easily.
