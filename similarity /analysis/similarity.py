import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keystroke_io import load_all  # noqa: E402

# ---------------------------------------------------------------- parameters
PAUSE_MAX_MS = 1000.0       # DD > 1 s inside a message = thinking/reading, not rhythm
MIN_SHARED = 5              # n-graphs two windows must share to be compared
ITEM_MIN_PER_PERSON = 10    # an n-graph is kept if every participant typed it >= 10 times
SIM_RATIO = 1.25            # "A" measure: two durations are similar if ratio <= 1.25
QUANTILES = np.linspace(0.05, 0.95, 19)
WINDOW_LENGTHS = [25, 50, 100, 200, 400]
REF_WINDOW = 100
DISTANCES = ["global", "ngraph", "ngraph_z", "R", "A", "R+A", "quantile"]
SHIFT_KEYS = {"ShiftLeft", "ShiftRight", "CapsLock"}

# validated categorical palette (fixed order, never cycled)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


# ================================================================ 1. latencies
def add_latencies(ks, pause_max=PAUSE_MAX_MS):
    """Add the digraph ending at each keystroke: DD, UD and its name 'KeyT>KeyH'.

    A digraph is only valid inside one sample (message) and when DD <= pause_max:
    the gap between two messages, or a long reading pause, is not a typing rhythm.
    """
    ks = ks.sort_values(["participant", "session", "down_ms"]).reset_index(drop=True)
    g = ks.groupby(["participant", "session"], sort=False)
    prev_code = g["key_code"].shift()
    same_sample = ks["sample"].eq(g["sample"].shift())
    dd = ks["down_ms"] - g["down_ms"].shift()
    ud = ks["down_ms"] - g["up_ms"].shift()
    valid = same_sample & dd.between(0, pause_max)
    ks["dd_ms"] = dd.where(valid)
    ks["ud_ms"] = ud.where(valid)
    ks["digraph"] = (prev_code + ">" + ks["key_code"]).where(valid)
    ks["pause"] = same_sample & (dd > pause_max)
    ks["transition"] = same_sample
    return ks


# ================================================================ 2. windows
def make_windows(ks, length, step=None):
    """Cut each (participant, session) keystroke stream into windows of `length` keys.

    Windows run across message boundaries (a message is often shorter than the
    window) but no digraph crosses a boundary (see add_latencies).
    Non-overlapping by default, so two windows never share keystrokes.
    """
    step = step or length
    windows = []
    for (p, s), g in ks.groupby(["participant", "session"], sort=True):
        idx = g.index.to_numpy()
        for k, start in enumerate(range(0, len(idx) - length + 1, step)):
            windows.append({"participant": p, "session": s, "window": k,
                            "rows": idx[start:start + length]})
    return windows


def global_features(w):
    hold = w["hold_ms"].to_numpy(float)
    dd = w["dd_ms"].dropna().to_numpy(float)
    ud = w["ud_ms"].dropna().to_numpy(float)
    q = lambda x, a: float(np.percentile(x, a)) if len(x) else np.nan  # noqa: E731
    n_tr = max(int(w["transition"].sum()), 1)
    return {
        "hold_median": q(hold, 50), "hold_iqr": q(hold, 75) - q(hold, 25),
        "dd_median": q(dd, 50), "dd_iqr": q(dd, 75) - q(dd, 25),
        "ud_median": q(ud, 50),
        "overlap_rate": float((ud < 0).mean()) if len(ud) else np.nan,
        "pause_rate": float(w["pause"].sum()) / n_tr,
        "backspace_rate": float((w["key_code"] == "Backspace").mean()),
        "shift_rate": float(w["key_code"].isin(SHIFT_KEYS).mean()),
    }


def build_vocabulary(ks, min_per_person=ITEM_MIN_PER_PERSON):
    """n-graphs typed at least `min_per_person` times by EVERY participant.

    Unigrams 'H:KeyE' (hold time) and digraphs 'DD:KeyT>KeyH' (press-press latency).
    Requiring every participant avoids items that only exist for one person
    (their presence alone would identify them, but would say nothing about rhythm).
    """
    uni = ks.groupby(["key_code", "participant"]).size().unstack(fill_value=0)
    dig = ks.dropna(subset=["digraph"]).groupby(["digraph", "participant"]).size() \
            .unstack(fill_value=0)
    keep_u = uni.index[(uni >= min_per_person).all(axis=1)]
    keep_d = dig.index[(dig >= min_per_person).all(axis=1)]
    return ["H:" + k for k in keep_u] + ["DD:" + d for d in keep_d]


def represent(ks, windows, vocab):
    """-> meta DataFrame, global matrix G, n-graph matrix V (log ms, NaN = absent),
          and log hold / log DD quantile vectors Q."""
    col = {item: j for j, item in enumerate(vocab)}
    V = np.full((len(windows), len(vocab)), np.nan)
    G, Q, meta = [], [], []
    for i, win in enumerate(windows):
        w = ks.loc[win["rows"]]
        G.append(global_features(w))
        logh = np.log(w["hold_ms"].clip(lower=1).to_numpy(float))
        logdd = np.log(w["dd_ms"].dropna().clip(lower=1).to_numpy(float))
        Q.append(np.concatenate([np.quantile(logh, QUANTILES),
                                 np.quantile(logdd, QUANTILES) if len(logdd) else
                                 np.full(len(QUANTILES), np.nan)]))
        for k, v in w.groupby("key_code")["hold_ms"].median().items():
            j = col.get("H:" + k)
            if j is not None:
                V[i, j] = np.log(max(v, 1))
        for d, v in w.dropna(subset=["digraph"]).groupby("digraph")["dd_ms"].median().items():
            j = col.get("DD:" + d)
            if j is not None:
                V[i, j] = np.log(max(v, 1))
        meta.append({"participant": win["participant"], "session": win["session"],
                     "window": win["window"], "n_keys": len(w),
                     "day": w["day"].iloc[0],
                     "duration_s": (w["down_ms"].iloc[-1] - w["down_ms"].iloc[0]) / 1000})
    G = pd.DataFrame(G)
    return pd.DataFrame(meta), G, V, np.array(Q)


# ================================================================ 3. distances
def zscore(X):
    X = np.asarray(X, float)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Z = (X - mu) / sd
    return np.where(np.isnan(Z), 0.0, Z)          # missing -> population mean


def _pairwise_masked(V, func, min_shared, chunk=64):
    """Generic chunked pairwise mean of func(|Vi - Vj|) over items both rows have."""
    n = len(V)
    present = ~np.isnan(V)
    Vf = np.where(present, V, 0.0)
    D = np.full((n, n), np.nan)
    shared_counts = np.zeros((n, n), dtype=int)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        both = present[a:b, None, :] & present[None, :, :]          # (c, n, K)
        cnt = both.sum(axis=2)
        val = func(np.abs(Vf[a:b, None, :] - Vf[None, :, :]))
        s = np.where(both, val, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[a:b] = np.where(cnt >= min_shared, s / cnt, np.nan)
        shared_counts[a:b] = cnt
    return D, shared_counts


def dist_global(G):
    Z = zscore(G)
    return np.abs(Z[:, None, :] - Z[None, :, :]).mean(axis=2)


def dist_ngraph(V, min_shared=MIN_SHARED, normalise=False):
    if normalise:
        sd = np.nanstd(V, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        V = V / sd
    return _pairwise_masked(V, lambda x: x, min_shared)


def dist_A(Vd, min_shared=MIN_SHARED, ratio=SIM_RATIO):
    t = np.log(ratio)
    D, _ = _pairwise_masked(Vd, lambda x: (x > t).astype(float), min_shared)
    return D


def disorder(a, b):
    """Gunetti-Picardi R: normalised disorder between the duration rankings of a and b."""
    n = len(a)
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable")
    max_dis = (n * n) / 2 if n % 2 == 0 else (n * n - 1) / 2
    return float(np.abs(ra - rb).sum() / max_dis) if max_dis else 0.0


def dist_R(Vd, min_shared=MIN_SHARED):
    n = len(Vd)
    present = ~np.isnan(Vd)
    D = np.full((n, n), np.nan)
    np.fill_diagonal(D, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            m = present[i] & present[j]
            if m.sum() >= min_shared:
                D[i, j] = D[j, i] = disorder(Vd[i, m], Vd[j, m])
    return D


def dist_quantile(Q):
    return np.nanmean(np.abs(Q[:, None, :] - Q[None, :, :]), axis=2)


def all_distances(G, V, Q, vocab, min_shared=MIN_SHARED):
    is_dig = np.array([v.startswith("DD:") for v in vocab])
    Vd = V[:, is_dig]
    D = {"global": dist_global(G)}
    D["ngraph"], shared = dist_ngraph(V, min_shared)
    D["ngraph_z"], _ = dist_ngraph(V, min_shared, normalise=True)
    D["R"] = dist_R(Vd, min_shared)
    D["A"] = dist_A(Vd, min_shared)
    D["R+A"] = D["R"] + D["A"]
    D["quantile"] = dist_quantile(Q)
    shared_dig = (~np.isnan(Vd)).astype(int) @ (~np.isnan(Vd)).astype(int).T
    return D, shared, shared_dig


# ================================================================ 4. evaluation
def auc(genuine, impostor):
    """P(genuine distance < impostor distance), ties count 1/2 (Mann-Whitney)."""
    g, i = np.asarray(genuine), np.asarray(impostor)
    if len(g) == 0 or len(i) == 0:
        return np.nan
    allv = np.concatenate([g, i])
    ranks = pd.Series(allv).rank().to_numpy()
    r_imp = ranks[len(g):].sum()
    return (r_imp - len(i) * (len(i) + 1) / 2) / (len(g) * len(i))


def eer(genuine, impostor):
    if len(genuine) == 0 or len(impostor) == 0:
        return np.nan
    th = np.unique(np.concatenate([genuine, impostor]))
    g, i = np.sort(genuine), np.sort(impostor)
    frr = 1 - np.searchsorted(g, th, side="right") / len(g)   # genuine above threshold
    far = np.searchsorted(i, th, side="right") / len(i)       # impostor at/below threshold
    k = np.argmin(np.abs(frr - far))
    return float((frr[k] + far[k]) / 2)


def pair_labels(meta):
    p = meta["participant"].to_numpy()
    s = (meta["participant"] + "_" + meta["session"]).to_numpy()
    iu = np.triu_indices(len(meta), k=1)
    same_p = p[iu[0]] == p[iu[1]]
    same_s = s[iu[0]] == s[iu[1]]
    return iu, same_p, same_s


def same_day_pairs(meta):
    d = (meta["participant"] + "_" + meta["day"].astype(str)).to_numpy()
    iu = np.triu_indices(len(meta), k=1)
    return d[iu[0]] == d[iu[1]]


def nn_accuracy(D, meta):
    """Leave-one-session-out 1-NN identification."""
    p = meta["participant"].to_numpy()
    s = (meta["participant"] + "_" + meta["session"]).to_numpy()
    hits, n = 0, len(meta)
    for i in range(n):
        d = np.where(s == s[i], np.nan, D[i])
        if np.all(np.isnan(d)):
            continue
        hits += p[np.nanargmin(d)] == p[i]
    return hits / n


def evaluate(D, meta):
    iu, same_p, same_s = pair_labels(meta)
    same_d = same_day_pairs(meta)
    rows = []
    for name, M in D.items():
        d = M[iu]
        ok = ~np.isnan(d)
        gen_x = d[ok & same_p & ~same_s]
        gen_s = d[ok & same_s]
        gen_d = d[ok & same_p & ~same_d]           # same person, another DAY
        imp = d[ok & ~same_p]
        rows.append({
            "distance": name, "coverage": ok.mean(),
            "auc_cross_session": auc(gen_x, imp), "eer_cross_session": eer(gen_x, imp),
            "auc_cross_day": auc(gen_d, imp), "eer_cross_day": eer(gen_d, imp),
            "auc_same_session": auc(gen_s, imp), "nn_accuracy": nn_accuracy(M, meta),
            "median_genuine_cross": np.median(gen_x) if len(gen_x) else np.nan,
            "median_impostor": np.median(imp) if len(imp) else np.nan,
            "n_genuine_cross": len(gen_x), "n_genuine_cross_day": len(gen_d),
            "n_impostor": len(imp),
        })
    return pd.DataFrame(rows)


def discriminability(meta, G, V, vocab):
    """Fisher ratio per feature: variance between participant means / mean within variance."""
    X = pd.concat([G.reset_index(drop=True),
                   pd.DataFrame(V, columns=vocab)], axis=1)
    X["participant"] = meta["participant"].to_numpy()
    rows = []
    for c in X.columns.drop("participant"):
        sub = X[[c, "participant"]].dropna()
        grp = sub.groupby("participant")[c]
        if grp.size().min() < 3:
            continue
        between = grp.mean().var(ddof=0)
        within = grp.var(ddof=1).mean()
        rows.append({"feature": c, "fisher_ratio": between / within if within > 0 else np.nan,
                     "windows_with_value": len(sub) / len(X)})
    return pd.DataFrame(rows).sort_values("fisher_ratio", ascending=False)


# ================================================================ 5. figures
def classical_mds(D, k=2):
    D = np.where(np.isnan(D), np.nanmax(D), D)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:k]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))


def fig_coverage(cov, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(cov["window"], cov["median_shared_digraphs"], marker="o", color=PALETTE[0])
    axes[0].set(xscale="log", xlabel="Window length (keystrokes)",
                ylabel="Median digraphs shared by 2 windows",
                title="Shared vocabulary grows with window length")
    axes[1].plot(cov["window"], 100 * cov["comparable_pairs"], marker="o", color=PALETTE[0])
    axes[1].set(xscale="log", xlabel="Window length (keystrokes)",
                ylabel=f"% of pairs sharing ≥ {MIN_SHARED} digraphs", ylim=(0, 105),
                title="Pairs that can be compared")
    for ax in axes:
        ax.set_xticks(cov["window"], labels=cov["window"])
        ax.minorticks_off()
    fig.tight_layout()
    fig.savefig(out / "coverage_vs_window.png")
    plt.close(fig)


def fig_metric_vs_window(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for c, name in enumerate(DISTANCES):
        s = summary[summary.distance == name]
        axes[0].plot(s["window"], s["auc_cross_session"], marker="o", ms=5,
                     color=PALETTE[c], label=name)
        axes[1].plot(s["window"], s["nn_accuracy"], marker="o", ms=5,
                     color=PALETTE[c], label=name)
    n_p = summary.attrs.get("n_participants", 4)
    axes[0].axhline(0.5, color=MUTED, lw=1, ls="--")
    axes[0].text(summary.window.min(), 0.51, "chance", color=MUTED, fontsize=8)
    axes[1].axhline(1 / n_p, color=MUTED, lw=1, ls="--")
    axes[1].text(summary.window.min(), 1 / n_p + 0.01, "chance", color=MUTED, fontsize=8)
    axes[0].set(ylabel="AUC (genuine pairs from other sessions)", ylim=(0.4, 1.0),
                title="Separation of same vs different person")
    axes[1].set(ylabel="1-NN accuracy (leave one session out)", ylim=(0, 1.0),
                title="Who typed this window?")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sorted(summary.window.unique()), labels=sorted(summary.window.unique()))
        ax.minorticks_off()
        ax.set_xlabel("Window length (keystrokes)")
    axes[1].legend(title="Distance", frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "auc_nn_vs_window.png")
    plt.close(fig)


def fig_histograms(D, meta, window, out):
    iu, same_p, same_s = pair_labels(meta)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, name in zip(axes.flat, DISTANCES):
        d = D[name][iu]
        ok = ~np.isnan(d)
        bins = np.linspace(np.nanpercentile(d, 0.5), np.nanpercentile(d, 99.5), 35)
        ax.hist(d[ok & ~same_p], bins=bins, density=True, color=PALETTE[1], alpha=0.55,
                label="different person")
        ax.hist(d[ok & same_p & ~same_s], bins=bins, density=True, color=PALETTE[0],
                alpha=0.55, label="same person, other session")
        ax.set_title(name)
        ax.set_yticks([])
        ax.grid(False)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Distance distributions, windows of {window} keystrokes", y=1.0)
    fig.tight_layout()
    fig.savefig(out / f"distance_histograms_w{window}.png")
    plt.close(fig)


def fig_block_heatmap(D, meta, name, window, out):
    lab = (meta["participant"] + "_" + meta["session"]).to_numpy()
    blocks = sorted(np.unique(lab))
    M = np.zeros((len(blocks), len(blocks)))
    for a, ba in enumerate(blocks):
        for b, bb in enumerate(blocks):
            sub = D[np.ix_(lab == ba, lab == bb)]
            if a == b:
                sub = sub[~np.eye(len(sub), dtype=bool)]
            M[a, b] = np.nanmean(sub)
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = ax.imshow(M, cmap="Blues_r")
    ax.set_xticks(range(len(blocks)), labels=blocks, rotation=90)
    ax.set_yticks(range(len(blocks)), labels=blocks)
    ax.grid(False)
    for a in range(len(blocks)):
        for b in range(len(blocks)):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[a, b] < (M.min() + M.max()) / 2 else INK)
    fig.colorbar(im, ax=ax, label=f"mean {name} distance (darker = closer)")
    ax.set_title(f"Mean '{name}' distance between sessions (windows of {window})")
    fig.tight_layout()
    fig.savefig(out / f"session_heatmap_{name}_w{window}.png")
    plt.close(fig)


def fig_mds(D, meta, name, window, out):
    Z = classical_mds(D)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    markers = {"S1": "o", "S2": "s", "S3": "^"}
    for c, p in enumerate(sorted(meta.participant.unique())):
        for s in sorted(meta.session.unique()):
            m = ((meta.participant == p) & (meta.session == s)).to_numpy()
            if m.any():
                ax.scatter(Z[m, 0], Z[m, 1], s=34, color=PALETTE[c],
                           marker=markers.get(s, "o"), edgecolor="white", linewidth=0.8,
                           label=f"{p} {s}")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    ax.set(xlabel="MDS 1", ylabel="MDS 2",
           title=f"Windows of {window} keystrokes, classical MDS on '{name}' distance")
    fig.tight_layout()
    fig.savefig(out / f"mds_{name}_w{window}.png")
    plt.close(fig)


def fig_discriminability(disc, window, out, top=20):
    d = disc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d["feature"], d["fisher_ratio"], color=PALETTE[0], height=0.65)
    ax.set(xlabel="Fisher ratio (between-person / within-person variance)",
           title=f"Most discriminative features, windows of {window}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"discriminative_features_w{window}.png")
    plt.close(fig)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Section 3.4: window representation and similarity")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--windows", type=int, nargs="+", default=WINDOW_LENGTHS)
    ap.add_argument("--ref-window", type=int, default=REF_WINDOW,
                    help="window length used for the detailed figures")
    ap.add_argument("--pause-max", type=float, default=PAUSE_MAX_MS)
    ap.add_argument("--min-shared", type=int, default=MIN_SHARED)
    ap.add_argument("--keep-test-phrase", action="store_true",
                    help="keep the '.tie5Roanl' collector-test samples (removed by default)")
    ap.add_argument("--out", default="results/similarity")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ks, report = load_all(args.csv, drop_test_phrase=not args.keep_test_phrase)
    ks = add_latencies(ks, args.pause_max)
    vocab = build_vocabulary(ks)
    n_uni = sum(v.startswith("H:") for v in vocab)
    print(f"\nVocabulary: {n_uni} keys (hold) + {len(vocab) - n_uni} digraphs (DD), "
          f"each typed >= {ITEM_MIN_PER_PERSON} times by every participant")
    pd.Series(vocab, name="item").to_csv(out / "vocabulary.csv", index=False)

    windows_list = sorted(set(args.windows) | {args.ref_window})
    summaries, coverage = [], []
    for L in windows_list:
        wins = make_windows(ks, L)
        meta, G, V, Q = represent(ks, wins, vocab)
        D, shared, shared_dig = all_distances(G, V, Q, vocab, args.min_shared)
        iu = np.triu_indices(len(meta), k=1)
        res = evaluate(D, meta)
        res.insert(0, "window", L)
        res.insert(1, "n_windows", len(meta))
        summaries.append(res)
        coverage.append({"window": L, "n_windows": len(meta),
                         "median_window_duration_s": meta.duration_s.median(),
                         "median_shared_digraphs": float(np.median(shared_dig[iu])),
                         "median_shared_items": float(np.median(shared[iu])),
                         "comparable_pairs": float((shared_dig[iu] >= args.min_shared).mean())})
        print(f"window {L:4d}: {len(meta):4d} windows, "
              f"median {coverage[-1]['median_window_duration_s']:.0f} s of typing, "
              f"median shared digraphs {coverage[-1]['median_shared_digraphs']:.0f}")
        if L == args.ref_window:
            ref = (meta, G, V, D)

    summary = pd.concat(summaries, ignore_index=True)
    summary.attrs["n_participants"] = ks.participant.nunique()
    cov = pd.DataFrame(coverage)
    summary.round(4).to_csv(out / "similarity_summary.csv", index=False)
    cov.round(4).to_csv(out / "coverage.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nSIMILARITY EVALUATION (genuine = same person, different session)")
    print(summary[["window", "distance", "coverage", "auc_cross_session",
                   "eer_cross_session", "auc_cross_day", "auc_same_session", "nn_accuracy"]]
          .round(3).to_string(index=False))

    meta, G, V, D = ref
    best = (summary[summary.window == args.ref_window]
            .sort_values("auc_cross_session", ascending=False).distance.iloc[0])
    disc = discriminability(meta, G, V, vocab)
    disc.round(4).to_csv(out / f"discriminative_features_w{args.ref_window}.csv", index=False)
    print(f"\nTop discriminative features (window {args.ref_window}):")
    print(disc.head(12).round(3).to_string(index=False))

    fig_coverage(cov, out)
    fig_metric_vs_window(summary, out)
    fig_histograms(D, meta, args.ref_window, out)
    fig_block_heatmap(D[best], meta, best, args.ref_window, out)
    fig_mds(D[best], meta, best, args.ref_window, out)
    fig_discriminability(disc, args.ref_window, out)
    (out / "data_report.txt").write_text("\n".join(report) + "\n")
    print(f"\nBest distance at window {args.ref_window}: {best}. Results in {out}/")


if __name__ == "__main__":
    main()
