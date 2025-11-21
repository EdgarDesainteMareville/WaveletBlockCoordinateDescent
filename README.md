# WaveletBlockCoordinateDescent
A flexible Python implementation of Block Coordinate Descent algorithms on Wavelet blocks for image reconstruction.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Minimal Example
A minimal example of how to use the Block Coordinate Descent implementation can be found in `minimal.py`. This script demonstrates how to set up the problem, run the algorithm, and evaluate the results.

### Demo Notebook
A small demo notebook is also provided in `demo.ipynb`.

### Adding your own update rules

To add your own update rules, you can add a function `create_my_update_list` in the `UpdateList` class in `multilevel/block.py`. The update list is the list
$$[(\varepsilon_{a_J}^k, \varepsilon_{d_J}^k, \ldots, \varepsilon_{d_1}^k), \quad 1 \leq k \leq K],$$
where $K$ is the total length of a cycle.

The variables $\varepsilon_{a_J}$ and $\varepsilon_{d_j}$ are encoded as follows:
- $\varepsilon_{a_J}$: `(0, 'approx')`
- $\varepsilon_{d_j}$: `(level, 'detail')` where `level` is $J - j$.

## Acknowledgements

This code uses the [DeepInverse library](https://deepinv.github.io/deepinv/) for inverse problems in imaging, and the [LazyLinOp library](https://faustgrp.gitlabpages.inria.fr/lazylinop/).