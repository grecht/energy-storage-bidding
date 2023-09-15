This repository displays the implementation and results of my [master's thesis](https://doi.org/10.5281/zenodo.8192258).
The implementation is in the directory `src/`, the experiments can be found in `exp/` including a notebook `results.ipynb` showing the reproducible results.

Building on [Jiang and Powell](https://pubsonline.informs.org/doi/10.1287/ijoc.2015.0640), I model the problem of bidding into the NYISO real-time market as an energy storage operator.
I use a simpler backward approximate dynamic programming approach with a scenario lattice to determine a near-optimal policy, i.e. a decision rule for placing bids.
In stylized experiments testing the approximation quality, the method performs just as well as in Jiang and Powell while leading to a speedup of 500-1000x in computation time on rather weak hardware.
However, it was not yet compared to a state of the art method in a more realistic setting.

![](fig/badp-lattice_sim_small_example.png)
