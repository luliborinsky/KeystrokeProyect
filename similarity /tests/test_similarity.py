"""Sanity checks for the distances of analysis/similarity.py.

Run:  python -m pytest tests/      (or simply: python tests/test_similarity.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from similarity import auc, disorder, dist_A, dist_ngraph, dist_R, eer  # noqa: E402


def test_disorder_identical_and_reversed():
    a = np.array([100.0, 150, 200, 250, 300])
    assert disorder(a, a) == 0.0
    assert disorder(a, a[::-1]) == 1.0          # completely reversed order
    assert disorder(a, a * 2) == 0.0            # R ignores global speed


def test_ngraph_distance_is_log_ratio():
    V = np.log(np.array([[100.0, 200, 300, 400, 500],
                         [200.0, 400, 600, 800, 1000]]))
    D, shared = dist_ngraph(V, min_shared=5)
    assert np.isclose(D[0, 1], np.log(2))       # everything twice slower
    assert shared[0, 1] == 5 and D[0, 0] == 0


def test_not_enough_shared_items_gives_nan():
    V = np.log(np.array([[100.0, 200, np.nan, np.nan],
                         [100.0, np.nan, 300, 400]]))
    D, _ = dist_ngraph(V, min_shared=2)
    assert np.isnan(D[0, 1])
    assert np.isnan(dist_R(V, min_shared=2)[0, 1])


def test_A_counts_similar_durations():
    V = np.log(np.array([[100.0, 100, 100, 100],
                         [110.0, 120, 200, 300]]))   # 2 within ratio 1.25, 2 not
    assert np.isclose(dist_A(V, min_shared=4)[0, 1], 0.5)


def test_auc_and_eer():
    assert auc([1, 2, 3], [4, 5, 6]) == 1.0
    assert auc([4, 5, 6], [1, 2, 3]) == 0.0
    assert eer(np.array([1.0, 2, 3]), np.array([4.0, 5, 6])) == 0.0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
