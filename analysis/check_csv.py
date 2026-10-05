#!/usr/bin/env python3
"""check_csv.py — Quick validation and summary of a keystroke CSV file.

Usage:
    python analysis/check_csv.py data/P07_S1.csv [--plot]

Checks:
- Header matches the common format exactly.
- Counts of down/up events.
- Unmatched presses (down without a paired up).
- Duplicate press_ids.
- Hold time statistics (overall and per frequent key).
- Press-press (DD) latency statistics.
- Per-sample breakdown: key count and reconstructed text.
- Optional --plot: timeline of one sample (horizontal bars per key).
"""

import sys
import argparse
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    "participant_id", "session_id", "sample_id",
    "event", "key_code", "timestamp_ms", "press_id", "key",
]


def validate_header(df):
    """Check that columns match the expected format exactly."""
    if list(df.columns) != EXPECTED_COLUMNS:
        print("❌ HEADER MISMATCH")
        print(f"   Expected: {EXPECTED_COLUMNS}")
        print(f"   Got:      {list(df.columns)}")
        return False
    print("✅ Header matches common format.")
    return True


def count_events(df):
    """Print counts of down and up events."""
    downs = df[df["event"] == "down"]
    ups = df[df["event"] == "up"]
    print(f"\n📊 Event counts:")
    print(f"   Downs: {len(downs)}")
    print(f"   Ups:   {len(ups)}")
    return downs, ups


def check_pairing(downs, ups):
    """Check for unmatched presses and duplicate press_ids."""
    down_ids = set(downs["press_id"])
    up_ids = set(ups["press_id"])

    unmatched = down_ids - up_ids
    print(f"\n🔗 Pairing:")
    print(f"   Unmatched downs (no matching up): {len(unmatched)}")
    if len(unmatched) > 0 and len(unmatched) <= 20:
        print(f"   IDs: {sorted(unmatched)}")

    orphan_ups = up_ids - down_ids
    if orphan_ups:
        print(f"   ⚠ Orphan ups (up without down): {len(orphan_ups)}")

    # Duplicate press_ids within downs or ups.
    dup_downs = downs[downs["press_id"].duplicated(keep=False)]
    dup_ups = ups[ups["press_id"].duplicated(keep=False)]
    if len(dup_downs) > 0:
        print(f"   ⚠ Duplicate press_ids in downs: {len(dup_downs)}")
    if len(dup_ups) > 0:
        print(f"   ⚠ Duplicate press_ids in ups: {len(dup_ups)}")
    if len(dup_downs) == 0 and len(dup_ups) == 0:
        print("   ✅ No duplicate press_ids.")


def hold_time_stats(downs, ups):
    """Compute hold times (up_timestamp - down_timestamp for paired events)."""
    down_ts = downs.set_index("press_id")["timestamp_ms"]
    up_ts = ups.set_index("press_id")["timestamp_ms"]

    common = down_ts.index.intersection(up_ts.index)
    if len(common) == 0:
        print("\n⏱ Hold times: no paired events.")
        return None

    holds = up_ts[common] - down_ts[common]
    print(f"\n⏱ Hold times (ms) — {len(holds)} paired events:")
    print(f"   Mean:   {holds.mean():.1f}")
    print(f"   Median: {holds.median():.1f}")
    print(f"   Std:    {holds.std():.1f}")
    print(f"   Min:    {holds.min():.1f}")
    print(f"   Max:    {holds.max():.1f}")

    negative = (holds < 0).sum()
    if negative:
        print(f"   ⚠ {negative} negative hold times (data issue).")

    # Per-key stats for the most frequent keys.
    down_keys = downs.set_index("press_id")["key_code"]
    key_holds = pd.DataFrame({"hold": holds, "key_code": down_keys[common]})
    freq = key_holds["key_code"].value_counts().head(10)
    print(f"\n   Top 10 keys by frequency:")
    for code in freq.index:
        subset = key_holds[key_holds["key_code"] == code]["hold"]
        print(f"     {code:20s}  n={len(subset):4d}  "
              f"mean={subset.mean():6.1f}  med={subset.median():6.1f}  "
              f"std={subset.std():5.1f}")

    return holds


def press_press_stats(downs):
    """Compute press-press (DD) latencies."""
    ts = downs.sort_values("timestamp_ms")["timestamp_ms"].values
    if len(ts) < 2:
        print("\n⏱ Press-press (DD): not enough data.")
        return
    dd = np.diff(ts)
    print(f"\n⏱ Press-press latency (DD, ms) — {len(dd)} intervals:")
    print(f"   Mean:   {dd.mean():.1f}")
    print(f"   Median: {np.median(dd):.1f}")
    print(f"   Std:    {dd.std():.1f}")
    print(f"   Min:    {dd.min():.1f}")
    print(f"   Max:    {dd.max():.1f}")


def per_sample_summary(df):
    """Per-sample breakdown: number of keys and reconstructed text."""
    downs = df[df["event"] == "down"]
    samples = downs.groupby("sample_id")

    print(f"\n📝 Per-sample summary ({len(samples)} samples):")
    for sid, group in samples:
        keys = group.sort_values("timestamp_ms")["key"].tolist()
        n = len(keys)
        # Reconstruct text from single-character keys.
        text_chars = []
        for k in keys:
            if k == "Backspace":
                if text_chars:
                    text_chars.pop()
            elif k == "Enter":
                text_chars.append("↵")
            elif len(k) == 1:
                text_chars.append(k)
            # skip modifier keys, arrows, etc.
        text = "".join(text_chars)
        print(f"   Sample {sid:3d}: {n:3d} keys → \"{text}\"")


def plot_sample(df, sample_id=None):
    """Plot a timeline of one sample: horizontal bar per key from press to release."""
    import matplotlib.pyplot as plt

    downs = df[df["event"] == "down"]
    ups = df[df["event"] == "up"]

    if sample_id is None:
        sample_id = downs["sample_id"].iloc[0]

    sample_downs = downs[downs["sample_id"] == sample_id].sort_values("timestamp_ms")
    down_ts = sample_downs.set_index("press_id")["timestamp_ms"]
    down_keys = sample_downs.set_index("press_id")["key"]

    up_ts = ups.set_index("press_id")["timestamp_ms"]
    common = down_ts.index.intersection(up_ts.index)

    fig, ax = plt.subplots(figsize=(14, max(3, len(common) * 0.25)))
    t0 = down_ts[common].min()

    for i, pid in enumerate(common):
        start = down_ts[pid] - t0
        end = up_ts[pid] - t0
        key = down_keys[pid]
        ax.barh(i, end - start, left=start, height=0.6, color="#e94560", alpha=0.8)
        ax.text(start - 5, i, key, ha="right", va="center", fontsize=8)

    ax.set_xlabel("Time (ms from first press)")
    ax.set_ylabel("Key (press order)")
    ax.set_title(f"Sample {sample_id} — key hold timeline")
    ax.invert_yaxis()
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Validate a keystroke CSV file.")
    parser.add_argument("csvfile", help="Path to the CSV file")
    parser.add_argument("--plot", action="store_true",
                        help="Plot a timeline of the first sample")
    args = parser.parse_args()

    print(f"═══ Checking: {args.csvfile} ═══\n")

    try:
        df = pd.read_csv(args.csvfile)
    except Exception as e:
        print(f"❌ Could not read CSV: {e}")
        sys.exit(1)

    if not validate_header(df):
        sys.exit(1)

    downs, ups = count_events(df)
    check_pairing(downs, ups)
    hold_time_stats(downs, ups)
    press_press_stats(downs)
    per_sample_summary(df)

    if args.plot:
        plot_sample(df)

    print("\n═══ Done ═══")


if __name__ == "__main__":
    main()
