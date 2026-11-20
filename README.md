# Triplet Node Influence (TNI)

Code and data for the paper **"Critical node identification through triplet node influence in complex networks"**, published in *Data Mining and Knowledge Discovery*.

TNI ranks the nodes of a network by combining three measures into a single score:

1. **Local influence**: node degree (or weighted degree, i.e. strength, for weighted networks).
2. **Gravity**: a gravity-style measure over nodes within distance `R` (`baseline/hvgc.py`).
3. **Global communicability**: the sum of a node's communicability to all other nodes, via the matrix exponential of the adjacency matrix. In weighted networks the adjacency matrix is degree-normalised first.

```
TNI(v) = LI(v) * Gravity(v) * GI(v) / N
```

where `N` is the number of nodes.

## Citation

If you use this code or data, please cite:

> Malang, C., & Phaphuangwittayakul, A. (2026). Critical node identification through triplet node influence in complex networks. *Data Mining and Knowledge Discovery*, 40(6), 113. https://doi.org/10.1007/s10618-026-01285-w

Paper: <https://link.springer.com/article/10.1007/s10618-026-01285-w>

BibTeX:

```bibtex
@article{malang2026tni,
  author  = {Malang, Chommaphat and Phaphuangwittayakul, Aniwat},
  title   = {Critical node identification through triplet node influence in complex networks},
  journal = {Data Mining and Knowledge Discovery},
  year    = {2026},
  volume  = {40},
  number  = {6},
  pages   = {113},
  doi     = {10.1007/s10618-026-01285-w},
  url     = {https://link.springer.com/article/10.1007/s10618-026-01285-w}
}
```

## Repository layout

| Path | Description |
|------|-------------|
| `tni.py` | TNI implementation (`compute_TNI`) |
| `utils.py` | Helper functions |
| `baseline/` | Baseline methods (`hvgc.py`, `glc.py`, `tpks.py`) and `compute_baseline.py` |
| `data/unweighted/`, `data/weighted/` | Network edge lists |
| `sir_ranking_*_results/` | SIR-based ground-truth rankings (CSV) |
| `sir_monotonicity_*_results/` | Monotonicity results from SIR experiments (`.dat`) |
| `compare_SIR_unweighted_baseline.ipynb` | Comparison with baselines on unweighted networks |
| `compare_SIR_weighted_baseline.ipynb` | Comparison with baselines on weighted networks |

## Usage

Requires Python 3. Install the dependencies with:

```bash
pip install -r requirements.txt
```

```python
import networkx as nx
from tni import compute_TNI

G = nx.read_edgelist("data/unweighted/02_Dolphins.edgelist", nodetype=int)
R = 2  # distance threshold for the gravity component
_, tni_scores = compute_TNI(G, R, network_type="unweighted")

ranking = sorted(tni_scores, key=tni_scores.get, reverse=True)
print(ranking[:10])
```

For weighted networks, load the graph with edge weights and pass `network_type="weighted"`.
To reproduce the experiments, run the two `compare_SIR_*_baseline.ipynb` notebooks.

## Datasets

- **Unweighted:** Karate, Dolphins, Polbooks, Jazz, WM2, Power, EEC, Email, Blogs, Router, PG, WikiVote
- **Weighted:** USAir97, NetScience, LesMis
