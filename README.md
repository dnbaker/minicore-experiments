This repository contains experiments for `minicore: Fast scRNA-seq clustering with various distances`.



### Dependencies
1. sklearn
2. numpy
3. scipy
4. minicore v0.3

### Installing minicore
To install minicore's Python bindings, clone [minicore recursively](https://github.com/dnbaker/minicore), change into the Python directory and run `python3 setup.py install`.
Alternatively, a single-command install would be `python3 -m pip install git+git://github.com/dnbaker/minicore@main`. Please create an issue if you encounter any difficulties.

Because compilation can be somewhat slow, the build is parallelized by default with `$OMP_NUM_THREADS` threads. Run your command as `OMP_NUM_THREADS=number python3 setup.py install` to use `number` threads.


### Setting up data

Data can be downloaded in compressed CSR-format [here](https://doi.org/10.5281/zenodo.4738365).
For simplicity, simply run this [download script](https://raw.githubusercontent.com/dnbaker/minicore-experiments/release/download.sh),
which will download and decompress the tarballs necessary.
It will prompt you to append `MINICORE_DATA` to your .bashrc; if you choose not to, you simply need to export `export MINICORE_DATA=${PATH}` to the path where you downloaded minicore\_data. Then you should be able to perform these experiments.
