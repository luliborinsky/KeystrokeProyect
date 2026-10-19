#!/usr/bin/env python3
"""person_P04.py
  1. load P04's windows (cleaning, windows and features from explore.py grom our github).
  2. build the models from P04:
       - profile  : distance to the user's profile (median, spread of each measure)
       - knn      : distance to the k closest windows of the user
       - outliers : number of measures far from the profile (outlier count)
       - ocsvm    : one-class model (One-Class SVM)
  3. score new windows: higher score = more unusual for P04.
  4. explain a score: which measures deviate from P04's profile

run:  python person_P04.py   (from the project folder or analysis/)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

# settlings

NAME = "P04"

# data the model is built
TRAIN_FILES = ["data/P04_S01.csv", "data/P04_S02.csv"]

# new windows to scor
NEW_FILES = ["data/P04_S03.csv"]

WINDOW = 30      # keystrokes per window (as in explore.py)
STEP = 15        # shift between windows
K = 5            # neighbours for kNN: small enough for little data
                 # large enough not to depend on one single window
OUTLIER_Z = 3.0  # a measure is an outlier if it is more than 3 spreads away

# folder of the project (contains data/). found automatically if this file is
# in the project folder or in analysis/
_HERE = Path(__file__).resolve().parent
PROJECT_DIR = _HERE if (_HERE / "data").is_dir() else _HERE.parent

sys.path[:0] = [str(_HERE), str(PROJECT_DIR / "analysis")]
from explore import read_and_clean, make_windows  # noqa: E402

META_COLS = ["participant", "session", "sample", "source_file", "window_start"]

# data

def load_windows(files):

    parts = []
    for f in map(Path, files):
        f = f if f.is_absolute() else PROJECT_DIR / f
        if not f.exists():
            print(f"  ! {f} not found, skipped")
            continue
        parts.append(read_and_clean(f))
    if not parts:
        raise SystemExit(f"No data found. Looked in {PROJECT_DIR / 'data'}")
    return make_windows(parts, WINDOW, STEP)


def to_X(windows, feats):
    return windows[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)

# models
# each model: fit(X) on the person's own windows. score(X) -> one number per window, the higher the more unusual.


class Scaler:
    """profile of each measure: centre = median, spread = 1.4826 * MAD
    (a robust standard deviation, not broken by a few strange windows)."""

    def fit(self, X):
        self.center = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - self.center), axis=0)
        spread = np.where(mad > 1e-9, mad, X.std(axis=0))
        self.scale = np.where(spread > 1e-9, spread, 1.0)
        return self

    def z(self, X):
        """how many spreads each measure is away from the median"""
        return (X - self.center) / self.scale


class Profile:
    """distance to the user's profile: mean over measures of |x - median| / spread."""
    name = "profile"

    def fit(self, X):
        self.sc = Scaler().fit(X)
        return self

    def score(self, X):
        return np.abs(self.sc.z(X)).mean(axis=1)


class KNN:
    """Mean distance to the K closest windows of the user.
    Similarity as in explore.py: standardised measures + Euclidean distance."""
    name = "knn"

    def fit(self, X):
        self.sc = Scaler().fit(X)
        self.T = self.sc.z(X)
        return self

    def score(self, X):
        Z = self.sc.z(X)
        d = np.sqrt(((Z[:, None, :] - self.T[None, :, :]) ** 2).sum(axis=2))
        return np.sort(d, axis=1)[:, :min(K, len(self.T))].mean(axis=1)


class Outliers:
    """outlier count: number of measures more than OUTLIER_Z spreads away"""
    name = "outliers"

    def fit(self, X):
        self.sc = Scaler().fit(X)
        return self

    def score(self, X):
        return (np.abs(self.sc.z(X)) > OUTLIER_Z).sum(axis=1).astype(float)


class OCSVM:
    """one-class model: one-class SVM on the standardised measures"""
    name = "ocsvm"

    def fit(self, X):
        self.sc = Scaler().fit(X)
        self.m = OneClassSVM(nu=0.1, gamma="scale").fit(self.sc.z(X))
        return self

    def score(self, X):
        return -self.m.decision_function(self.sc.z(X))


METHODS = [Profile, KNN, Outliers, OCSVM]

# main


def main():
    pd.set_option("display.width", 140)

    # data of the person.
    train = load_windows(TRAIN_FILES)
    feats = [c for c in train.columns if c not in META_COLS]
    X = to_X(train, feats)
    print(f"=== {NAME}: model built from {len(X)} own windows "
          f"({WINDOW} keys, step {STEP}) ===\n")

    # build the models from the person's own data only
    models = {M.name: M().fit(X) for M in METHODS}

    profile = models["profile"].sc
    prof = pd.DataFrame({"median": profile.center, "spread": profile.scale}, index=feats)
    print(f"Profile of {NAME} (median and spread of each measure):")
    print(prof.round(2).to_string(), "\n")

    # score new windows
    new = load_windows(NEW_FILES)
    Xn = to_X(new, feats)
    scores = pd.DataFrame({name: m.score(Xn) for name, m in models.items()})
    scores.insert(0, "session", new["session"].to_numpy())
    scores.insert(1, "sample", new["sample"].to_numpy())
    scores.insert(2, "window_start", new["window_start"].to_numpy())

    print(f"Scores of {len(Xn)} new windows (higher = more unusual for {NAME}):")
    print(scores.head(10).round(2).to_string(index=False))
    print("  ...\n")
    print("Summary of the new scores:")
    print(scores[[m.name for m in METHODS]].describe().loc[
        ["mean", "50%", "min", "max"]].round(2).to_string(), "\n")

    # 4. Explain: why is a window unusual? Which measures deviate.
    i = int(np.argmax(scores["profile"]))
    z = profile.z(Xn[i:i + 1])[0]
    top = np.argsort(-np.abs(z))[:3]
    print(f"Most unusual new window (sample={scores['sample'][i]}, "
          f"window_start={scores['window_start'][i]}, "
          f"profile score={scores['profile'][i]:.2f}):")
    for j in top:
        print(f"  {feats[j]:14s} value={Xn[i, j]:8.2f}   usual={profile.center[j]:8.2f}"
              f"   -> {z[j]:+.1f} spreads")

    out = PROJECT_DIR / "results"
    out.mkdir(exist_ok=True)
    prof.to_csv(out / f"{NAME}_profile.csv")
    scores.to_csv(out / f"{NAME}_scores.csv", index=False)
    print(f"\nSaved results/{NAME}_profile.csv and results/{NAME}_scores.csv")


if __name__ == "__main__":
    main()
