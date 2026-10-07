

import argparse
from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


MAX_FILES = 5
DEFAULT_HOLD_MAX = 2_000.0       # ms
DEFAULT_PAUSE_MAX = 2_000.0      # ms
DEFAULT_WINDOW = 30
DEFAULT_STEP = 15


def subtitle(ax_or_fig, participants):
    names = sorted(map(str, participants))
    text = "Participants: " + ", ".join(names)
    fig = ax_or_fig if isinstance(ax_or_fig, plt.Figure) else ax_or_fig.figure
    fig.text(0.5, 0.955, text, ha="center", va="top", fontsize=10)


def read_and_clean(path, hold_max=DEFAULT_HOLD_MAX, pause_max=DEFAULT_PAUSE_MAX):
    df = pd.read_csv(path)

    required = {"event", "timestamp_ms", "press_id", "key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}")

    if "participant_id" not in df.columns:
        df["participant_id"] = Path(path).stem.split("_")[0]
    if "session_id" not in df.columns:
        df["session_id"] = Path(path).stem
    if "sample_id" not in df.columns:
        df["sample_id"] = 1

    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", "press_id"]).copy()
    df["press_id"] = df["press_id"].astype(str)
    df["key"] = df["key"].fillna("").astype(str)

    # Normalize correction keys.
    correction_keys = {
        "backspace", "back", "delete", "del",
        "back_space", "backspace_key"
    }
    is_correction = df["key"].str.lower().isin(correction_keys)

    downs = df[df["event"].str.lower().eq("down")].copy()
    ups = df[df["event"].str.lower().eq("up")].copy()

    # If there is more than one down/up event for a press_id, keep
    # the first down and the first subsequent up.
    downs = downs.sort_values("timestamp_ms").drop_duplicates("press_id", keep="first")
    ups = ups.sort_values("timestamp_ms").drop_duplicates("press_id", keep="first")

    pairs = downs.merge(
        ups[["press_id", "timestamp_ms"]],
        on="press_id",
        suffixes=("_down", "_up"),
        how="inner",
    )
    pairs["duration_ms"] = pairs["timestamp_ms_up"] - pairs["timestamp_ms_down"]
    pairs = pairs[pairs["duration_ms"] >= 0].copy()

    # Keep sample/session/participant information from the key-down event.
    pairs = pairs.rename(columns={"timestamp_ms_down": "down_ms",
                                  "timestamp_ms_up": "up_ms"})

    # Correction keys: remove completely.
    pairs = pairs[~pairs["key"].str.lower().isin(correction_keys)].copy()

    # Keystrokes held too long: remove.
    pairs = pairs[pairs["duration_ms"] <= hold_max].copy()

    pairs = pairs.sort_values(["participant_id", "session_id", "sample_id", "down_ms"])
    pairs["source_file"] = Path(path).name
    pairs["file_id"] = Path(path).stem

    # Time since the previous key-down within the same sample.
    group_cols = ["participant_id", "session_id", "sample_id"]
    pairs["inter_down_ms"] = pairs.groupby(group_cols)["down_ms"].diff()

    # Inter-key time, excluding long pauses.
    pairs["valid_interkey"] = pairs["inter_down_ms"].between(0, pause_max, inclusive="both")

    # Detect overlaps: a new key goes down before the previous one is released.
    prev_up = pairs.groupby(group_cols)["up_ms"].shift(1)
    pairs["overlap"] = pairs["down_ms"] < prev_up

    return pairs


def summarize_cleaning(raw_path, clean):
    raw = pd.read_csv(raw_path)
    raw_pairs = raw[raw["event"].astype(str).str.lower().eq("down")].shape[0]
    removed = raw_pairs - len(clean)
    print(f"\n[{Path(raw_path).name}]")
    print(f"  original key-down events: {raw_pairs:,}")
    print(f"  valid keystrokes:   {len(clean):,}")
    print(f"  removed:             {removed:,}")
    print(f"  overlaps:             {int(clean['overlap'].sum()):,}")
    if len(clean):
        print(f"  median duration:      {clean['duration_ms'].median():.1f} ms")
        print(f"  mean duration:        {clean['duration_ms'].mean():.1f} ms")


def choose_sample(clean, sample_id=None):
    if clean.empty:
        return clean
    if sample_id is None:
        return clean.sort_values("sample_id").iloc[:].copy()
    subset = clean[clean["sample_id"].astype(str) == str(sample_id)].copy()
    if subset.empty:
        print(f"  Warning: sample_id={sample_id} not found. Using the first available sample.")
        first = clean["sample_id"].iloc[0]
        subset = clean[clean["sample_id"] == first].copy()
    return subset


def plot_timeline(clean, sample_id=None):
    sample = choose_sample(clean, sample_id)
    if sample.empty:
        return

    # To keep the figure manageable, show at most 80 keystrokes.
    sample = sample.sort_values("down_ms").head(80).copy()
    t0 = sample["down_ms"].min()
    sample["start_s"] = (sample["down_ms"] - t0) / 1000
    sample["end_s"] = (sample["up_ms"] - t0) / 1000

    fig, ax = plt.subplots(figsize=(14, 6))
    y = np.arange(len(sample))

    for yi, (_, row) in zip(y, sample.iterrows()):
        width = max(row["end_s"] - row["start_s"], 0.001)
        ax.plot([row["start_s"], row["end_s"]], [yi, yi], linewidth=7,
                solid_capstyle="butt")
        label = row["key"] if row["key"] != " " else "␠"
        ax.text(row["start_s"] + width / 2, yi, label,
                ha="center", va="center", fontsize=8)

        if row["overlap"]:
            ax.plot(row["start_s"], yi, marker="x", markersize=9,
                    markeredgewidth=2)

    ax.set_xlabel("Time from message start (s)")
    ax.set_ylabel("Keystroke order")
    ax.set_title("Keystroke Timeline — overlaps marked with ×")
    subtitle(ax, sample["participant_id"].unique())
    ax.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def plot_histograms(all_clean):
    if not all_clean:
        return

    # Duraciones.
    fig, ax = plt.subplots(figsize=(11, 6))
    participants = sorted({str(x) for d in all_clean for x in d["participant_id"].unique()})
    for d in all_clean:
        for p, g in d.groupby("participant_id"):
            ax.hist(g["duration_ms"], bins=40, alpha=0.35, density=True,
                    label=str(p))
    ax.set_xlabel("Keystroke duration (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Keystroke Duration Distribution")
    subtitle(ax, participants)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    # Inter-key.
    fig, ax = plt.subplots(figsize=(11, 6))
    for d in all_clean:
        for p, g in d.groupby("participant_id"):
            vals = g.loc[g["valid_interkey"], "inter_down_ms"].dropna()
            vals = vals[vals >= 0]
            if len(vals):
                ax.hist(vals, bins=40, alpha=0.35, density=True, label=str(p))
    ax.set_xlabel("Time between consecutive keystrokes (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Inter-Key Time Distribution")
    subtitle(ax, participants)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def digraph_counts(all_clean):
    rows = []
    for d in all_clean:
        for (p, s, sample), g in d.groupby(
            ["participant_id", "session_id", "sample_id"], sort=False
        ):
            keys = g.sort_values("down_ms")["key"].tolist()
            for a, b in zip(keys, keys[1:]):
                rows.append((str(p), a, b))

    if not rows:
        print("\nNot enough data to compute digraphs.")
        return pd.DataFrame()

    dig = pd.DataFrame(rows, columns=["participant", "k1", "k2"])
    dig["digraph"] = dig["k1"] + "→" + dig["k2"]
    table = dig.groupby(["participant", "digraph"]).size().reset_index(name="count")
    table = table.sort_values(["participant", "count"], ascending=[True, False])

    print("\n=== TOP DIGRAPHS POR PSEUDÓNIMO ===")
    for p, g in table.groupby("participant"):
        print(f"\n{p}")
        print(g.head(15).to_string(index=False))

    # Visualización: top global, separado por usuario.
    top = table.groupby("digraph")["count"].sum().nlargest(20).index
    plot = table[table["digraph"].isin(top)].pivot_table(
        index="digraph", columns="participant", values="count", fill_value=0
    )
    if not plot.empty:
        ax = plot.sort_values(plot.columns.tolist(), ascending=False).head(20).plot(
            kind="bar", figsize=(13, 6)
        )
        ax.set_xlabel("Digraph")
        ax.set_ylabel("Number of occurrences")
        ax.set_title("Most Frequent Digraphs")
        subtitle(ax, table["participant"].unique())
        ax.grid(axis="y", alpha=0.2)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    return table


def feature_vector(g):
    """Compact feature vector comparable across windows."""
    dur = g["duration_ms"].to_numpy(float)
    inter = g.loc[g["valid_interkey"], "inter_down_ms"].dropna().to_numpy(float)

    def stats(x, prefix):
        if len(x) == 0:
            return {f"{prefix}_{k}": 0.0 for k in
                    ["mean", "std", "median", "q10", "q90", "iqr"]}
        return {
            f"{prefix}_mean": np.mean(x),
            f"{prefix}_std": np.std(x),
            f"{prefix}_median": np.median(x),
            f"{prefix}_q10": np.percentile(x, 10),
            f"{prefix}_q90": np.percentile(x, 90),
            f"{prefix}_iqr": np.percentile(x, 75) - np.percentile(x, 25),
        }

    out = {}
    out.update(stats(dur, "dur"))
    out.update(stats(inter, "inter"))
    out["overlap_rate"] = float(g["overlap"].mean())
    out["space_rate"] = float((g["key"] == " ").mean())
    return out


def make_windows(all_clean, window=DEFAULT_WINDOW, step=DEFAULT_STEP):
    rows = []
    for d in all_clean:
        for (p, s, sample), g in d.groupby(
            ["participant_id", "session_id", "sample_id"], sort=False
        ):
            g = g.sort_values("down_ms").reset_index(drop=True)
            if len(g) < window:
                continue
            for start in range(0, len(g) - window + 1, step):
                w = g.iloc[start:start + window]
                feat = feature_vector(w)
                feat.update({
                    "participant": str(p),
                    "session": str(s),
                    "sample": sample,
                    "source_file": w["source_file"].iloc[0],
                    "window_start": start,
                })
                rows.append(feat)
    return pd.DataFrame(rows)


def plot_2d_windows(windows):
    if windows.empty:
        print("\nNot enough windows for PCA/MDS.")
        return
    if not SKLEARN_OK:
        print("\nInstall scikit-learn for PCA/MDS: pip install scikit-learn")
        return

    meta_cols = {"participant", "session", "sample", "source_file", "window_start"}
    feature_cols = [c for c in windows.columns if c not in meta_cols]
    X = windows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove constant features.
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] < 1 or len(X) < 2:
        return

    Xz = StandardScaler().fit_transform(X)

    # PCA.
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xz)

    fig, ax = plt.subplots(figsize=(10, 7))
    for p in sorted(windows["participant"].unique()):
        mask = windows["participant"].eq(p)
        ax.scatter(Z[mask, 0], Z[mask, 1], alpha=0.65, s=35, label=p)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Keystroke Windows Projected with PCA")
    subtitle(ax, windows["participant"].unique())
    ax.legend(title="Participant")
    ax.grid(alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    # MDS can be expensive for many points; limit to 250.
    n = min(len(Xz), 250)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xz), n, replace=False) if len(Xz) > n else np.arange(len(Xz))
    Xm = Xz[idx]

    mds = MDS(n_components=2, random_state=42, n_init=2,
              max_iter=500, normalized_stress="auto")
    Zm = mds.fit_transform(Xm)

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = windows["participant"].iloc[idx].to_numpy()
    for p in sorted(np.unique(labels)):
        mask = labels == p
        ax.scatter(Zm[mask, 0], Zm[mask, 1], alpha=0.65, s=35, label=p)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Keystroke Windows Projected with MDS")
    subtitle(ax, windows["participant"].unique())
    ax.legend(title="Participant")
    ax.grid(alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def compare_users(windows):
    if windows.empty:
        return

    participants = sorted(windows["participant"].unique())
    if len(participants) < 2:
        print("\n=== COMPARISON BETWEEN PARTICIPANTS ===")
        print("Only one participant is available: differences between participants cannot be measured.")
        return

    meta_cols = {"participant", "session", "sample", "source_file", "window_start"}
    feature_cols = [c for c in windows.columns if c not in meta_cols]
    X = windows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.loc[:, X.nunique() > 1]
    Xz = StandardScaler().fit_transform(X)

    centroids = {}
    for p in participants:
        centroids[p] = Xz[windows["participant"].eq(p)].mean(axis=0)

    print("\n=== DISTANCES BETWEEN PARTICIPANT CENTROIDS ===")
    rows = []
    for a, b in combinations(participants, 2):
        dist = np.linalg.norm(centroids[a] - centroids[b])
        rows.append({"participant_A": a, "participant_B": b, "distance": dist})
    print(pd.DataFrame(rows).sort_values("distance").to_string(index=False))


def stability_report(all_clean, windows):
    print("\n=== STABILITY BY PARTICIPANT ===")
    file_counts = {}
    for d in all_clean:
        for p in d["participant_id"].unique():
            file_counts.setdefault(str(p), set()).add(d["source_file"].iloc[0])

    for p, files in sorted(file_counts.items()):
        if len(files) < 2:
            print(f"{p}: insufficient — only {len(files)} file(s).")
            continue

        w = windows[windows["participant"].eq(p)].copy()
        if len(w) < 2:
            print(f"{p}: insufficient — not enough windows.")
            continue

        meta_cols = {"participant", "session", "sample", "source_file", "window_start"}
        feature_cols = [c for c in w.columns if c not in meta_cols]
        X = w[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X = X.loc[:, X.nunique() > 1]
        Xz = StandardScaler().fit_transform(X)

        # Distancias entre ventanas de sesiones/archivos distintos del mismo usuario.
        within = []
        by_file = {}
        for f in w["source_file"].unique():
            by_file[f] = Xz[w["source_file"].eq(f)]

        file_centroids = {f: arr.mean(axis=0) for f, arr in by_file.items()}
        for a, b in combinations(file_centroids, 2):
            within.append(np.linalg.norm(file_centroids[a] - file_centroids[b]))

        # Si existen otros usuarios, usamos su distance entre centroides como referencia.
        other = windows[windows["participant"] != p]
        between = []
        if not other.empty:
            all_cent = {}
            for q in other["participant"].unique():
                qx = other[other["participant"].eq(q)][feature_cols].fillna(0)
                qx = qx.loc[:, X.columns]
                # Reutilizamos escalado del usuario actual de forma aproximada.
                qz = StandardScaler().fit_transform(qx) if len(qx) > 1 else qx.to_numpy()
                all_cent[str(q)] = qz.mean(axis=0)
            # Comparing independently scaled features is not ideal; therefore
            # the final conclusion is based mainly on dispersion across files.
        med_within = float(np.median(within)) if within else np.nan

        # Coeficiente de variación de las medias por archivo para características positivas.
        per_file = w.groupby("source_file")[feature_cols].mean()
        cvs = []
        for col in feature_cols:
            m = per_file[col].mean()
            if abs(m) > 1e-9 and len(per_file) >= 2:
                cvs.append(abs(per_file[col].std(ddof=1) / m))
        median_cv = float(np.median(cvs)) if cvs else np.nan

        # Explicit operational rule:
        # estable = mediana de CV <= 0.30 y no hay una divergencia enorme entre archivos.
        stable = median_cv <= 0.30 if np.isfinite(median_cv) else False

        print(
            f"{p}: {'STABLE' if stable else 'VARIABLE'} | "
            f"files={len(files)}, "
            f"distance entre centroides (mediana)={med_within:.3f}, "
            f"median CV={median_cv:.3f}"
        )

    
    print("\nCriterion: a participant is considered stable if the median CV of the features")
    print("across files from the same participant is <= 0.30. This is an exploratory")
    print("rule, not a statistical identity test.")


def main():
    parser = argparse.ArgumentParser(description="EDA de dinámica de teclado")
    parser.add_argument("csv", nargs="+", help="1 to 5 files CSV")
    parser.add_argument("--sample-id", default=None,
                        help="sample/message for the timeline; first sample by default")
    parser.add_argument("--hold-max", type=float, default=DEFAULT_HOLD_MAX,
                        help="maximum key duration in ms (default 2000)")
    parser.add_argument("--pause-max", type=float, default=DEFAULT_PAUSE_MAX,
                        help="maximum inter-key pause considered in ms (default 2000)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="keystrokes per window (default 30)")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help="step between windows (default 15)")
    args = parser.parse_args()

    paths = [Path(x) for x in args.csv]
    if len(paths) > MAX_FILES:
        raise SystemExit(f"Máximo {MAX_FILES} archivos. Recibidos: {len(paths)}")
    for p in paths:
        if not p.exists():
            raise SystemExit(f"No existe: {p}")

    all_clean = []
    for p in paths:
        clean = read_and_clean(p, args.hold_max, args.pause_max)
        summarize_cleaning(p, clean)
        all_clean.append(clean)

    participants = sorted({
        str(p) for d in all_clean for p in d["participant_id"].unique()
    })
    print("\n=== DATASET ===")
    print(f"Files: {len(paths)}")
    print(f"Participants: {', '.join(participants)}")

    # Gráficos principales.
    plot_timeline(all_clean[0], args.sample_id)
    plot_histograms(all_clean)

    # Digraphs.
    digraph_counts(all_clean)

    # Ventanas + comparaciones.
    windows = make_windows(all_clean, args.window, args.step)
    print(f"\nWindows created: {len(windows)}")
    compare_users(windows)
    stability_report(all_clean, windows)
    plot_2d_windows(windows)

    print("\n=== DONE ===")
    print("Figures are displayed on screen and tables/summaries are printed to the console.")


if __name__ == "__main__":
    main()
