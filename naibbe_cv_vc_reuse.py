# Copyright (c) 2025, Michael A. Greshko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software, datasets, and associated documentation files (the "Software
# and Datasets"), to deal in the Software and Datasets without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software and Datasets, and to
# permit persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# - The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software and Datasets.
# - Any publications making use of the Software and Datasets, or any substantial
#   portions thereof, shall cite the Software and Datasets's original publication:
#
# > Greshko, Michael A. (2025). The Naibbe cipher: a substitution cipher that
#   encrypts Latin and Italian as Voynich Manuscript-like ciphertext.
#   Cryptologia. https://doi.org/10.1080/01611194.2025.2566408
#
# THE SOFTWARE AND DATASETS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE AND DATASETS.

import argparse
import csv
import math
import random
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================
# Parameters (default)
# =========================
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")
TABLES = ['alpha', 'beta1', 'beta2', 'beta3', 'gamma1', 'gamma2']
STATES = ['unigram', 'prefix', 'suffix']

# Respacing: 17=standard (slight bigram excess), 18=simplified (50/50)
RESPACING = 17

# Respacing mode: "random" (original), "cv" (consonant-vowel), "vc" (vowel-consonant)
RESPACING_MODE = "cv"
VOWELS = set("aeiou")

USE_78_CARD_DECK = False
SPACE_REMOVAL_RATE = 0.03
UNAMBIGUOUS = True
MAX_BIGRAM_RETRIES = 10000

CARD_WEIGHTS = {
    False: {'alpha': 20, 'beta1': 8, 'beta2': 8, 'beta3': 8, 'gamma1': 4, 'gamma2': 4},
    True:  {'alpha': 28, 'beta1': 14, 'beta2': 11, 'beta3': 11, 'gamma1': 7, 'gamma2': 7},
}

# ===== Reuse controls (independent) =====
# Short-range reuse: small-lag bump (i=1..9) via exponential lags
ENABLE_SHORT_RANGE_REUSE = True
SR_REUSE_RATE = 0.004      # prob per token to attempt SR reuse
SR_MAX_LAG = 9
SR_DECAY_TAU = 3.0

# Long-range reuse: heavy-tailed lags (Pareto), raises Hurst exponent
ENABLE_LONG_RANGE_REUSE = True
LR_REUSE_RATE = 0.006      # prob per token to attempt LR reuse
LR_TAIL_ALPHA = 0.75
LR_MAX_LAG = 20000

# =========================
# Stats / instrumentation
# =========================
REUSE_STATS = {
    "sr_attempts": 0,
    "sr_successes": 0,
    "lr_attempts": 0,
    "lr_successes": 0,
    "sr_lag_hist": Counter(),
    "lr_lag_hist": Counter(),
}
ambiguity_retries = 0  # bigram ambiguity retries

def reset_reuse_stats():
    REUSE_STATS["sr_attempts"] = 0
    REUSE_STATS["sr_successes"] = 0
    REUSE_STATS["lr_attempts"] = 0
    REUSE_STATS["lr_successes"] = 0
    REUSE_STATS["sr_lag_hist"].clear()
    REUSE_STATS["lr_lag_hist"].clear()

def print_reuse_stats():
    print("\n=== Reuse stats ===")
    print(f"Short-range attempts:  {REUSE_STATS['sr_attempts']}")
    print(f"Short-range successes: {REUSE_STATS['sr_successes']}")
    print(f"Long-range attempts:   {REUSE_STATS['lr_attempts']}")
    print(f"Long-range successes:  {REUSE_STATS['lr_successes']}")
    if REUSE_STATS['sr_lag_hist']:
        top_sr = REUSE_STATS['sr_lag_hist'].most_common(10)
        print(f"SR lag histogram (top 10): {top_sr}")
    if REUSE_STATS['lr_lag_hist']:
        top_lr = REUSE_STATS['lr_lag_hist'].most_common(10)
        print(f"LR lag histogram (top 10): {top_lr}")

def dump_reuse_stats_csv(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "metric", "value"])
        w.writerow(["sr", "attempts", REUSE_STATS["sr_attempts"]])
        w.writerow(["sr", "successes", REUSE_STATS["sr_successes"]])
        w.writerow(["lr", "attempts", REUSE_STATS["lr_attempts"]])
        w.writerow(["lr", "successes", REUSE_STATS["lr_successes"]])
        w.writerow([])
        w.writerow(["type", "lag", "count"])
        for lag, c in sorted(REUSE_STATS["sr_lag_hist"].items()):
            w.writerow(["sr", lag, c])
        for lag, c in sorted(REUSE_STATS["lr_lag_hist"].items()):
            w.writerow(["lr", lag, c])

# =========================
# Tables & glyphs
# =========================
def generate_placeholder_tables() -> Dict[str, Dict[Tuple[str, str], str]]:
    table = defaultdict(dict)
    for table_name in TABLES:
        for state in STATES:
            for letter in ALPHABET:
                key = f"{state}_{table_name}_{letter}"
                table[table_name][(state, letter)] = key
    return table

naibbe_tables = generate_placeholder_tables()

# Load glyph mappings
glyph_df = pd.read_csv("references/naibbe_tables.csv")  # must be present
placeholder_to_glyph = dict(zip(glyph_df['code'], glyph_df['glyphs']))

# Precompute unigram glyphs to avoid collisions
unigram_glyphs = {
    glyph for code, glyph in placeholder_to_glyph.items()
    if code.startswith("unigram_")
}

def build_bigram_catalog(alphabet, tables, glyph_map):
    """
    combined_str -> set of (prefix_code, suffix_code) pairs.
    Collision exists when len(set) > 1 for a combined_str.
    """
    bigram_catalog = defaultdict(set)
    prefix_map, suffix_map = {}, {}
    for t in tables:
        for L in alphabet:
            p_code = f"prefix_{t}_{L}"
            s_code = f"suffix_{t}_{L}"
            p_g = glyph_map.get(p_code, p_code)
            s_g = glyph_map.get(s_code, s_code)
            prefix_map[(t, L)] = (p_code, p_g)
            suffix_map[(t, L)] = (s_code, s_g)
    for t1, L1 in prefix_map:
        p_code, p_glyph = prefix_map[(t1, L1)]
        for t2, L2 in suffix_map:
            s_code, s_glyph = suffix_map[(t2, L2)]
            combined = p_glyph + s_glyph
            bigram_catalog[combined].add((p_code, s_code))
    return bigram_catalog

bigram_catalog = build_bigram_catalog(ALPHABET, TABLES, placeholder_to_glyph)

# =========================
# Deck creation
# =========================
def create_card_deck(use_78=False) -> List[str]:
    weights = CARD_WEIGHTS[use_78]
    deck = []
    for table, count in weights.items():
        deck.extend([table] * count)
    random.shuffle(deck)
    return deck

# =========================
# Respacing
# =========================
def _is_vowel(ch: str) -> bool:
    return ch in VOWELS

def _wants_bigram_cv(a: str, b: str) -> bool:
    return (a not in VOWELS) and (b in VOWELS)

def _wants_bigram_vc(a: str, b: str) -> bool:
    return (a in VOWELS) and (b not in VOWELS)

def respace_plaintext(text: str, pre_plaintext_file=None) -> List[str]:
    """
    RESPACING_MODE:
      - 'random': original 17/36 or 18/36 behavior via RESPACING
      - 'cv': consonant-vowel bigrams only; else unigrams
      - 'vc': vowel-consonant bigrams only; else unigrams
    """
    t = text.lower().replace(" ", "")
    i = 0
    out = []

    if RESPACING_MODE == "random":
        while i < len(t):
            if i == len(t) - 1 or random.random() < (RESPACING / 36):
                out.append(t[i])
                i += 1
            else:
                out.append(t[i:i+2])
                i += 2

    elif RESPACING_MODE == "cv":
        while i < len(t):
            if i == len(t) - 1:
                out.append(t[i]); i += 1
            else:
                a, b = t[i], t[i+1]
                if _wants_bigram_cv(a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(a); i += 1

    elif RESPACING_MODE == "vc":
        while i < len(t):
            if i == len(t) - 1:
                out.append(t[i]); i += 1
            else:
                a, b = t[i], t[i+1]
                if _wants_bigram_vc(a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(a); i += 1
    else:
        raise ValueError(f"Unknown RESPACING_MODE: {RESPACING_MODE}")

    if pre_plaintext_file is not None:
        pre_plaintext_file.write(" ".join(out) + "\n")
    return out

# =========================
# Plaintext signature + reuse logic (aligned)
# =========================
def _plain_sig(token_str: str):
    """Canonical signature for plaintext token identity."""
    if len(token_str) == 1:
        return ('u', token_str)
    else:
        return ('b', token_str[0], token_str[1])

def _sample_exp_lag(max_lag: int, tau: float) -> int:
    if max_lag <= 0:
        return 1
    weights = [math.exp(-i / max(1e-9, tau)) for i in range(1, max_lag + 1)]
    s = sum(weights)
    r = random.random() * s
    acc = 0.0
    for i, w in enumerate(weights, start=1):
        acc += w
        if r <= acc:
            return i
    return max_lag

def _sample_pareto_lag(alpha: float, max_lag: int) -> int:
    if max_lag <= 0:
        return 1
    L = 1 + int(random.paretovariate(max(1e-6, alpha)))
    return max(1, min(L, max_lag))

def _find_matching_reuse_lag(plain_history: List[Tuple], L0: int, curr_sig) -> Optional[int]:
    """Search L0, then ±1, ±2, ... for a lag whose plaintext signature equals curr_sig."""
    n = len(plain_history)
    if n <= 0:
        return None

    def ok(L):
        if 1 <= L <= n:
            return plain_history[-L] == curr_sig
        return False

    if ok(L0):
        return L0

    max_radius = max(L0 - 1, n - L0)
    for r in range(1, max_radius + 1):
        left = L0 - r
        right = L0 + r
        left_ok = ok(left)
        right_ok = ok(right)
        if left_ok and right_ok:
            return left  # tie-breaker; change to random.choice([left, right]) if desired
        if left_ok:
            return left
        if right_ok:
            return right
    return None

def _maybe_choose_reuse_lag_with_alignment(
    plain_history: List[Tuple], curr_sig
) -> Tuple[Optional[int], Optional[str]]:
    """
    Try short-range reuse first (if enabled), aligned to a lag whose plaintext matches curr_sig.
    If that fails, try long-range reuse (if enabled), likewise aligned.
    Returns (lag, reuse_type) where reuse_type in {"SR","LR"} or (None, None).
    """
    hist_len = len(plain_history)
    if hist_len <= 0:
        return None, None

    # Short-range
    if ENABLE_SHORT_RANGE_REUSE:
        REUSE_STATS["sr_attempts"] += 1
        if random.random() < SR_REUSE_RATE:
            Lprop = _sample_exp_lag(min(SR_MAX_LAG, hist_len), SR_DECAY_TAU)
            Lmatch = _find_matching_reuse_lag(plain_history, Lprop, curr_sig)
            if Lmatch is not None:
                REUSE_STATS["sr_successes"] += 1
                REUSE_STATS["sr_lag_hist"][Lmatch] += 1
                return Lmatch, "SR"

    # Long-range
    if ENABLE_LONG_RANGE_REUSE:
        REUSE_STATS["lr_attempts"] += 1
        if random.random() < LR_REUSE_RATE:
            Lprop = _sample_pareto_lag(LR_TAIL_ALPHA, min(LR_MAX_LAG, hist_len))
            Lmatch = _find_matching_reuse_lag(plain_history, Lprop, curr_sig)
            if Lmatch is not None:
                REUSE_STATS["lr_successes"] += 1
                REUSE_STATS["lr_lag_hist"][Lmatch] += 1
                return Lmatch, "LR"

    return None, None

# =========================
# Encryption
# =========================
def encrypt_naibbe(
    plaintext: str,
    tables: Dict[str, Dict[Tuple[str, str], str]],
    glyph_map: Dict[str, str],
    use_78: bool = False,
    pre_plaintext_file=None,
    collect_reuse_log: Optional[List[Tuple[int, str, int]]] = None,  # (index, type, lag)
) -> List[str]:
    """
    Encrypt a normalized plaintext line. If reuse fires, the ciphertext token is copied
    verbatim from the aligned lag, and plaintext identity is preserved (by construction).
    """
    global ambiguity_retries, bigram_catalog

    ngrams = respace_plaintext(plaintext, pre_plaintext_file)
    ciphertext: List[str] = []
    plain_history: List[Tuple] = []

    deck = create_card_deck(use_78)
    deck_index = 0

    for idx, token in enumerate(ngrams):
        curr_sig = _plain_sig(token)

        # Conditional, plaintext-aligned reuse (SR first, then LR)
        L, rtype = _maybe_choose_reuse_lag_with_alignment(plain_history, curr_sig)
        if L is not None:
            ciphertext.append(ciphertext[-L])
            plain_history.append(curr_sig)
            if collect_reuse_log is not None and rtype is not None:
                collect_reuse_log.append((len(ciphertext)-1, rtype, L))
            continue

        # Otherwise, fresh generation
        if len(token) == 1:
            if deck_index >= len(deck):
                deck = create_card_deck(use_78); deck_index = 0
            table = deck[deck_index]; deck_index += 1
            code = tables[table][('unigram', token)]
            glyph = glyph_map.get(code, code)
            ciphertext.append(glyph)
            plain_history.append(curr_sig)

        else:
            a, b = token[0], token[1]
            if UNAMBIGUOUS:
                accepted = False
                for _ in range(MAX_BIGRAM_RETRIES):
                    if deck_index >= len(deck):
                        deck = create_card_deck(use_78); deck_index = 0
                    table_prefix = deck[deck_index]; deck_index += 1
                    code_prefix = tables[table_prefix][('prefix', a)]
                    glyph_prefix = glyph_map.get(code_prefix, code_prefix)

                    if deck_index >= len(deck):
                        deck = create_card_deck(use_78); deck_index = 0
                    table_suffix = deck[deck_index]; deck_index += 1
                    code_suffix = tables[table_suffix][('suffix', b)]
                    glyph_suffix = glyph_map.get(code_suffix, code_suffix)

                    combined = glyph_prefix + glyph_suffix

                    if combined in unigram_glyphs:
                        ambiguity_retries += 1; continue
                    pairs = bigram_catalog.get(combined, set())
                    if any(pair != (code_prefix, code_suffix) for pair in pairs):
                        ambiguity_retries += 1; continue

                    ciphertext.append(combined)
                    plain_history.append(curr_sig)
                    accepted = True
                    break

                if not accepted:
                    ciphertext.append(glyph_prefix + glyph_suffix)
                    plain_history.append(curr_sig)

            else:
                if deck_index >= len(deck):
                    deck = create_card_deck(use_78); deck_index = 0
                t1 = deck[deck_index]; deck_index += 1
                code1 = tables[t1][('prefix', a)]
                g1 = glyph_map.get(code1, code1)

                if deck_index >= len(deck):
                    deck = create_card_deck(use_78); deck_index = 0
                t2 = deck[deck_index]; deck_index += 1
                code2 = tables[t2][('suffix', b)]
                g2 = glyph_map.get(code2, code2)

                ciphertext.append(g1 + g2)
                plain_history.append(curr_sig)

    return ciphertext

# =========================
# Normalization & spacing
# =========================
def clean_line(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    no_diacritics = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    replacements = {
        'æ': 'ae', 'Æ': 'ae',
        'œ': 'oe', 'Œ': 'oe',
        'ð': 'd',  'Ð': 'd',
        'þ': 'th', 'Þ': 'th',
        'ł': 'l',  'Ł': 'l',
        'ß': 'ss',
        'ø': 'o',  'Ø': 'o'
    }
    replaced = ''.join(replacements.get(c, c) for c in no_diacritics)
    cleaned = ''.join(c for c in replaced if c.isalpha()).upper()
    cleaned = cleaned.replace("W", "UU").replace("J", "I").replace("K", "C")
    return cleaned.lower()

def respace_line(line: str, drop_rate: float) -> str:
    if drop_rate <= 0:
        return line.strip()
    if drop_rate >= 1:
        return line.replace(" ", "")
    tokens = line.strip().split()
    if len(tokens) < 2:
        return line.strip()
    out = tokens[0]
    for tok in tokens[1:]:
        if random.random() < drop_rate:
            out += tok
        else:
            out += ' ' + tok
    return out

# =========================
# Unit tests (quick checks)
# =========================
def _unit_test_plaintext_alignment():
    """
    Create a short plaintext, force respacing & both reuses on,
    and assert that every reuse event's plaintext matches current plaintext.
    """
    print("\n[TEST] plaintext alignment on reuse…")
    global RESPACING_MODE, ENABLE_SHORT_RANGE_REUSE, ENABLE_LONG_RANGE_REUSE
    saved_mode = RESPACING_MODE
    saved_sr = ENABLE_SHORT_RANGE_REUSE
    saved_lr = ENABLE_LONG_RANGE_REUSE

    RESPACING_MODE = "cv"  # deterministic-ish segmentation
    ENABLE_SHORT_RANGE_REUSE = True
    ENABLE_LONG_RANGE_REUSE = True

    reset_reuse_stats()
    reuse_log: List[Tuple[int, str, int]] = []

    pt = "ARMAVIRUMQUECANO" * 5  # repeated to ensure some matches in history
    cleaned = clean_line(pt)
    ct = encrypt_naibbe(cleaned, naibbe_tables, placeholder_to_glyph,
                        use_78=USE_78_CARD_DECK, pre_plaintext_file=None,
                        collect_reuse_log=reuse_log)

    # Reconstruct aligned plaintext signatures to verify equality on reuse
    # We re-run respacing_plaintext to get the segmentation actually used
    tokens = respace_plaintext(cleaned, pre_plaintext_file=None)
    plain_history: List[Tuple] = []
    emitted = 0
    errors = 0

    # We need to replay encrypt_naibbe's reuse decisions; since that requires RNG state,
    # instead we check post-hoc using the recorded reuse_log and the same _plain_sig mapping.
    for idx, (out_index, rtype, lag) in enumerate(reuse_log):
        # out_index corresponds to the emitted token index in ciphertext
        # Build plain_history up to that index
        while len(plain_history) < out_index:
            sig = _plain_sig(tokens[emitted])
            plain_history.append(sig)
            emitted += 1
        curr_sig = _plain_sig(tokens[emitted])
        src_sig = plain_history[-lag] if lag <= len(plain_history) else None
        if curr_sig != src_sig:
            errors += 1

        # Append the reused token's sig
        plain_history.append(curr_sig)
        emitted += 1

    assert errors == 0, f"Found {errors} reuse events with mismatched plaintext!"
    print("[TEST] ok – all reuse events preserved plaintext identity.")

    # restore
    RESPACING_MODE = saved_mode
    ENABLE_SHORT_RANGE_REUSE = saved_sr
    ENABLE_LONG_RANGE_REUSE = saved_lr

def _unit_test_basic_encrypt():
    print("\n[TEST] basic encrypt on a short line…")
    cleaned = clean_line("Lorem ipsum DOLOR sit amet, æther, coelum; jacta, König!")
    tokens = respace_plaintext(cleaned)
    ct = encrypt_naibbe(cleaned, naibbe_tables, placeholder_to_glyph)
    assert len(ct) == len(tokens)
    print("[TEST] ok – ciphertext token count matches respaced plaintext.")

# =========================
# CLI / main
# =========================
import sys
import argparse

def build_parser():
    p = argparse.ArgumentParser(description="Naibbe cipher generator with respacing modes and aligned reuse.", add_help=True)
    p.add_argument("--input", default="input/examples/nathist_book16.txt")
    p.add_argument("--output", default="encrypted/nathist_output_ciphertext_cv_srlr.txt")
    p.add_argument("--respaced-output", default="encrypted/nathist_output_ciphertext_respaced_cv_srlr.txt")
    p.add_argument("--pre-plaintext-out", default="encrypted/nathist_pre_encryption_respaced_plaintext_cv_srlr.txt")

    p.add_argument("--respacing-mode", choices=["random","cv","vc"], default=RESPACING_MODE)
    p.add_argument("--respacing", type=int, default=RESPACING)

    p.add_argument("--enable-sr", action="store_true", default=ENABLE_SHORT_RANGE_REUSE)
    p.add_argument("--sr-rate", type=float, default=SR_REUSE_RATE)
    p.add_argument("--sr-max-lag", type=int, default=SR_MAX_LAG)
    p.add_argument("--sr-tau", type=float, default=SR_DECAY_TAU)

    p.add_argument("--enable-lr", action="store_true", default=ENABLE_LONG_RANGE_REUSE)
    p.add_argument("--lr-rate", type=float, default=LR_REUSE_RATE)
    p.add_argument("--lr-alpha", type=float, default=LR_TAIL_ALPHA)
    p.add_argument("--lr-max-lag", type=int, default=LR_MAX_LAG)

    p.add_argument("--use-78", action="store_true", default=USE_78_CARD_DECK)
    p.add_argument("--unambiguous", action="store_true", default=UNAMBIGUOUS)
    p.add_argument("--space-drop", type=float, default=SPACE_REMOVAL_RATE)

    p.add_argument("--print-stats", action="store_true")
    p.add_argument("--dump-stats-csv", default="")
    p.add_argument("--run-tests", action="store_true")
    return p

def parse_args(argv=None):
    parser = build_parser()
    # Use parse_known_args so Jupyter's -f ... doesn't crash
    args, unknown = parser.parse_known_args(argv)
    # You can optionally log unknown but we'll silently ignore
    return args

def main(argv=None):
    args = parse_args(argv)
    if args.run_tests:
        _unit_test_basic_encrypt()
        _unit_test_plaintext_alignment()
        return
    reset_reuse_stats()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout, \
         open(args.respaced_output, "w", encoding="utf-8") as frespace, \
         open(args.pre_plaintext_out, "w", encoding="utf-8") as fplain:

        for line in fin:
            cleaned = clean_line(line)
            if cleaned:
                reuse_log: List[Tuple[int, str, int]] = []
                encrypted_tokens = encrypt_naibbe(
                    cleaned, naibbe_tables, placeholder_to_glyph,
                    use_78=USE_78_CARD_DECK,
                    pre_plaintext_file=fplain,
                    collect_reuse_log=reuse_log
                )
                line_out = " ".join(encrypted_tokens)
                fout.write(line_out + "\n")
                frespace.write(respace_line(line_out, SPACE_REMOVAL_RATE) + "\n")
            else:
                fout.write("\n")
                frespace.write("\n")
                fplain.write("\n")

    if UNAMBIGUOUS:
        print(f"Total ambiguity retries: {ambiguity_retries}")

    if args.print_stats or args.dump_stats_csv:
        print_reuse_stats()
        if args.dump_stats_csv:
            dump_reuse_stats_csv(args.dump_stats_csv)
            print(f"Stats written to: {args.dump_stats_csv}")

if __name__ == "__main__":
    main()
