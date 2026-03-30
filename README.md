# Upper-Level Ranking Approximation CMA-ES (URA-CMA-ES)

This code implements the Upper-Level Ranking Approximation CMA-ES (URA-CMA-ES)
algorithm described in the paper *Accelerating Black-Box Bilevel
Optimization with Rank-Based Upper-Level Value Function Approximation*.

## How to use 
We use `uv` as a dependency manager. The dependencies of this project can be
installed via `uv sync`.

The `BilevelCMA` class in `bilevel_cma.py` implements URA-CMA-ES, and
`run_experiment.py` implements the benchmarks on SMD and WRA problem suites.

## Acknowledgements 
The SMD problem suite `smd.py` is a Python reimplementation of the functions described in
[<a href="#ref1">1</a>]
.
The code for the dd-CMA-ES algorithm used within URA-CMA-ES is adapted 
from [<a href="#ref2">2</a>, <a href="#ref3">3</a> (paper)], and the
code for the WRA problem suite `journal_minmaxf.py` is adapted from [<a
href="#ref4">4</a>, <a href="#ref5">5</a> (paper)].

## References
<a id="ref1">[1]</a> Sinha, A., Malo, P., & Deb, K. (2014). Test problem construction for single-objective bilevel optimization. *Evolutionary Computation, 22*(3), 439-477.

<a id="ref2">[2]</a> `https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518`

<a id="ref3">[3]</a> Akimoto, Y., & Hansen, N. (2020). Diagonal acceleration for covariance matrix adaptation evolution strategies. *Evolutionary Computation, 28*(3), 405-435.

<a id="ref2">[4]</a> `https://github.com/akimotolab/worstcase-ranking-approximation`

<a id="ref3">[5]</a> Miyagi, A., Miyauchi, Y., Maki, A., Fukuchi, K., Sakuma, J., & Akimoto, Y. (2023). Covariance matrix adaptation evolutionary strategy with worst-case ranking approximation for min–max optimization and its application to berthing control tasks. *ACM Transactions on Evolutionary Learning, 3*(2), 1-32.
