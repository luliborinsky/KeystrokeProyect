"""keystroke_io.py — Robust loading of the raw keystroke CSVs into one keystroke table.

Turns raw down/up events (common class format) into one row per keystroke:

    participant, session, sample, key, key_code, down_ms, up_ms, hold_ms

and repairs the data-quality problems we found while exploring our own files:

1. Malformed rows: some rows for the comma key were written as one quoted field,
   (the whole record P01,S01,16,down,Comma,... ends up inside field 0)
   -> the inner record is parsed again so the comma keystroke is kept.
2. Duplicated recordings: a file may contain events already present in another
   file (P02_S01 is a byte-for-byte copy of P01_S01's events; P02_S03 also
   contains all of P02_S02). Events are de-duplicated on
   (timestamp_ms, press_id, event): the first file that contains an event keeps it.
   A file that loses almost all its events is reported as a duplicate and skipped.
3. Wrong session labels inside the file (P03_S03 says "S02"): participant and
   session are taken from the FILE NAME (P07_S1.csv), which is what the protocol fixes.
4. Collector test phrase: samples containing '.tie5Roanl' are fixed text, not
   free text, and are removed by default (drop_test_phrase=True).

Usage (quick data-quality report):
    python analysis/keystroke_io.py data/*.csv
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["participant_id", "session_id", "sample_id", "event",
           "key_code", "timestamp_ms", "press_id", "key"]
HOLD_MAX_MS = 2000.0          # a key held longer than 2 s is not normal typing
DUPLICATE_FILE_RATIO = 0.9    # file whose events are >=90 % already seen -> duplicate
TEST_PHRASE_RE = r"roanl|tie5|5roa"   # collector test phrase '.tie5Roanl' (with typos)
FILE_RE = re.compile(r"(P\d+)_S0*(\d+)", re.IGNORECASE)



def ids_from_filename(path):
    """Extract participant and session IDs from filenames."""
    filename = Path(path).stem
    m = re.search(r"(P\d+)_S0*(\d+)$", filename, re.IGNORECASE)

    if not m:
        raise ValueError(
            f"{path}: filename must look like P05_S01.csv or P07_S1.csv"
        )

    participant = m.group(1).upper()
    session = f"S{int(m.group(2))}"

    return participant, session


def read_raw(path):
    """Read one raw CSV with the csv module, repairing rows quoted as a single field."""
    rows, repaired, dropped = [], 0, 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if [h.strip() for h in header[:8]] != COLUMNS:
            raise ValueError(f"{path}: unexpected header {header}")
        for row in reader:
            if row and row[0].count(",") >= 7:          # whole record inside field 0
                row = next(csv.reader([row[0]]))
                repaired += 1
            if len(row) < 8:
                dropped += 1
                continue
            rows.append(row[:8])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["press_id"] = pd.to_numeric(df["press_id"], errors="coerce")
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce")
    df["event"] = df["event"].str.strip().str.lower()
    bad = df[["timestamp_ms", "press_id"]].isna().any(axis=1) | ~df["event"].isin(["down", "up"])
    dropped += int(bad.sum())
    return df[~bad].copy(), repaired, dropped



def normalize_key_code(code, key):
    """Normalize physical key codes across collector versions."""

    raw = "" if pd.isna(code) else str(code).strip()
    value = "" if pd.isna(key) else str(key)

    if raw in {"ShiftLeft", "ShiftRight", "16"}:
        return "Shift"

    if raw and not raw.isdigit() and raw not in {
        "Unidentified", "nan"
    }:
        return raw

    special_keys = {
        "Backspace": "Backspace",
        "Enter": "Enter",
        "CapsLock": "CapsLock",
        "Escape": "Escape",
        "Delete": "Delete",
        "ArrowLeft": "ArrowLeft",
        "ArrowRight": "ArrowRight",
        " ": "Space",
    }

    if value in special_keys:
        return special_keys[value]

    if len(value) == 1 and value.isalpha() and value.isascii():
        return "Key" + value.upper()

    if len(value) == 1 and value.isdigit():
        return "Digit" + value

    punctuation = {
        ",": "Comma",
        ".": "Period",
        "'": "Quote",
        ";": "Semicolon",
        "/": "Slash",
        "-": "Minus",
        "=": "Equal",
        "[": "BracketLeft",
        "\\": "Backslash",
        "`": "Backquote",
    }

    if value in punctuation:
        return punctuation[value]

    legacy_codes = {
        "8": "Backspace",
        "13": "Enter",
        "20": "CapsLock",
        "32": "Space",
        "37": "ArrowLeft",
        "39": "ArrowRight",
        "46": "Delete",
        "188": "Comma",
        "190": "Period",
    }

    if raw in legacy_codes:
        return legacy_codes[raw]

    if not raw or raw in {"Unidentified", "nan"}:
        return "Unidentified"

    return "LegacyKeyCode_" + raw

def to_keystrokes(df, participant, session, hold_max=HOLD_MAX_MS):
    """Pair each down with its up (press_id) -> one row per keystroke, in time order."""
    downs = df[df.event == "down"].sort_values("timestamp_ms").drop_duplicates("press_id")
    ups = df[df.event == "up"].sort_values("timestamp_ms").drop_duplicates("press_id")
    ks = downs.merge(ups[["press_id", "timestamp_ms"]], on="press_id",
                     suffixes=("_down", "_up"))
    ks = ks.rename(columns={"timestamp_ms_down": "down_ms", "timestamp_ms_up": "up_ms",
                            "sample_id": "sample"})
    ks["hold_ms"] = ks["up_ms"] - ks["down_ms"]
    n_unpaired = len(downs) - len(ks)
    ks = ks[(ks.hold_ms >= 0) & (ks.hold_ms <= hold_max)]
    ks["participant"] = participant
    ks["session"] = session
    # physical key (layout independent); fall back on the character if code is unknown
    ks["key_code"] = [
    normalize_key_code(code, key)
    for code, key in zip(ks["key_code"], ks["key"])
    ]
    ks = ks.sort_values("down_ms").reset_index(drop=True)
    cols = ["participant", "session", "sample", "key", "key_code",
            "down_ms", "up_ms", "hold_ms", "press_id"]
    return ks[cols], n_unpaired


def test_phrase_samples(ks, pattern=TEST_PHRASE_RE):
    """Boolean mask of keystrokes belonging to a sample that contains the test phrase."""
    chars = ks["key"].where(ks["key"].str.len() == 1, "")
    text = chars.groupby([ks["participant"], ks["session"], ks["sample"]]).transform(
        lambda s: "".join(s).lower())
    return text.str.contains(pattern, regex=True)


def load_all(paths, hold_max=HOLD_MAX_MS, drop_test_phrase=True, verbose=True):
    """Load several files -> (keystrokes DataFrame, list of report lines).

    drop_test_phrase: remove the samples where the collector test phrase
    '.tie5Roanl' was typed. It is FIXED text typed 10 times by everybody, so it is
    not free text and would make sessions look artificially similar.
    """
    seen = set()
    parts, report = [], []
    for path in sorted(map(str, paths)):
        participant, session = ids_from_filename(path)
        raw, repaired, dropped = read_raw(path)
        sig = list(zip(raw.timestamp_ms.round(1), raw.press_id, raw.event))
        new = np.array([s not in seen for s in sig], dtype=bool)
        dup_ratio = 1 - new.mean() if len(new) else 0.0
        line = f"{Path(path).name}: {len(raw)} events"
        if repaired:
            line += f", {repaired} malformed rows repaired"
        if dropped:
            line += f", {dropped} unreadable rows dropped"
        labels = sorted(raw.session_id.dropna().unique())
        if labels and labels != [session] and labels != [f"S{int(session[1:]):02d}"]:
            line += f", session column says {labels} (using {session} from file name)"
        if dup_ratio >= DUPLICATE_FILE_RATIO:
            report.append(line + f" -> SKIPPED: {dup_ratio:.0%} of events duplicate another file")
            continue
        if dup_ratio > 0:
            line += f", {int((~new).sum())} events duplicating another file removed"
        seen.update(s for s, keep in zip(sig, new) if keep)
        ks, unpaired = to_keystrokes(raw[new], participant, session, hold_max)
        if unpaired:
            line += f", {unpaired} unpaired downs dropped"
        if drop_test_phrase and len(ks):
            test = test_phrase_samples(ks)
            if test.any():
                line += f", {int(test.sum())} test-phrase keystrokes removed"
                ks = ks[~test].reset_index(drop=True)
        ks["day"] = pd.to_datetime(ks["down_ms"].min(), unit="ms").date().isoformat() \
            if len(ks) else ""
        line += f" -> {len(ks)} keystrokes"
        report.append(line)
        parts.append(ks)
    if verbose:
        print("DATA LOADING")
        for line in report:
            print("  " + line)
    return pd.concat(parts, ignore_index=True), report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python analysis/keystroke_io.py data/*.csv")
    ks, _ = load_all(sys.argv[1:])
    print()
    print(ks.groupby(["participant", "session"]).agg(
        day=("day", "first"), keystrokes=("key", "size"), samples=("sample", "nunique"),
        median_hold_ms=("hold_ms", "median")).to_string())
