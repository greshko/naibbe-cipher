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
#   encrypts Latin and Italian as Voynich Manuscript-like ciphertext. Preprint;
#   submitted to Cryptologia.
#   
# THE SOFTWARE AND DATASETS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE AND DATASETS.


"""
NOTE: The code in 'alphabet_code' is URL-safe base64 of compact JSON:
  {"v":1,"A1":[...26 glyphs...],"A2":[...],"A3":[...]}

To decode the alphabet code, in the event you want to check a given Voynichesque variant's cipher alphabets:
    import base64, json
    s = base64.urlsafe_b64decode(code + '===').decode()
    payload = json.loads(s)  # payload["A1"], payload["A2"], payload["A3"] in A..Z order
"""

import csv
import math
import random
import string
import json
import base64
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter, namedtuple
from typing import List, Tuple, Dict, Any, Optional

# ========= PATHS =========
CSV_PATH = "references/voynichesque_alphabet_options.csv"
LOG_CSV_PATH = "voynichesque_sweep_log.csv"

# ========= GLYPH CLUSTERS (metrics parsing only) =========
GLYPH_CLUSTERS = ["cfh", "ckh", "cph", "cth", "sh", "ch"]  # longest-first

# ========= DATA MODELS =========
Option = namedtuple("Option", ["code", "alphabet", "glyph", "length"])

@dataclass
class VoynParams:
    glyph_len_ratio: List[int]  # [r1, r2, r3, r4], sum=20
    u_prob: float
    t_prob: float
    b_prob: float
    prob_bigram_B: float
    prob_unigram_A2: float
    prob_null_y: float


# ========= CSV LOADING =========
def _normalize_alphabet_field(val) -> int:
    s = str(val).strip().lower()
    for k in ("alphabet", "alpha", "a"):
        s = s.replace(k, "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return int(digits)
    parts = s.split()
    for p in parts[::-1]:
        if p.isdigit():
            return int(p)
    raise ValueError(f"Cannot parse alphabet field: {val!r}")

def load_options(csv_path: str) -> List[Option]:
    options: List[Option] = []
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        headers = {h.lower(): h for h in reader.fieldnames}
        alph_col   = headers.get("alphabet")
        glyph_col  = headers.get("glyph") or headers.get("glyphs") or headers.get("string")
        length_col = headers.get("length") or headers.get("len")
        code_col   = headers.get("code") or list(reader.fieldnames)[0]
        if not (alph_col and glyph_col and length_col):
            raise ValueError("CSV missing required headers: 'alphabet', 'glyph', 'length'.")
        for row in reader:
            code   = (row.get(code_col, "") or "").strip()
            alph   = _normalize_alphabet_field(row[alph_col])
            glyph  = row[glyph_col].strip()
            length = int(row[length_col])
            options.append(Option(code, alph, glyph, length))
    return options


# ========= LENGTH QUOTAS FROM RATIOS =========
def quotas_from_ratio(ratio: List[int], total: int = 26) -> List[int]:
    if sum(ratio) <= 0:
        raise ValueError("glyph_len_ratio must have positive sum.")
    base = [r / sum(ratio) * total for r in ratio]
    floors = [math.floor(x) for x in base]
    remainder = total - sum(floors)
    fracs = sorted([(base[i] - floors[i], i) for i in range(len(base))],
                   key=lambda t: (-t[0], t[1]))
    for _, i in fracs[:remainder]:
        floors[i] += 1
    return floors  # sums exactly to total


# ========= ALPHABET BUILDING =========
def build_cipher_alphabet(options: List[Option], alphabet_id: int, length_ratio: List[int]) -> Dict[str, str]:
    pool = [opt for opt in options if opt.alphabet == alphabet_id]
    bins: Dict[int, List[str]] = defaultdict(list)
    for opt in pool:
        if 1 <= opt.length <= 4:
            bins[opt.length].append(opt.glyph)

    quotas = quotas_from_ratio(length_ratio, total=26)   # lengths [1,2,3,4]
    for L, needed in enumerate(quotas, start=1):
        have = len(bins[L])
        if have < needed:
            raise ValueError(
                f"Not enough options for Alphabet {alphabet_id}, length {L}: need {needed}, have {have}."
            )

    selected: List[str] = []
    for L, needed in enumerate(quotas, start=1):
        ranked = sorted(bins[L], key=lambda _: random.random())
        selected.extend(ranked[:needed])

    ranks = sorted(((random.random(), g) for g in selected))
    ranked_glyphs = [g for _, g in ranks]
    letters = list(string.ascii_uppercase)  # A..Z
    return dict(zip(letters, ranked_glyphs))


# ========= VOYNICHESQUE PIPELINE =========
alphabet1: Dict[str, str] = {}
alphabet2: Dict[str, str] = {}
alphabet3_base: Dict[str, str] = {}
alphabet3: Dict[str, str] = {}
alphabet3_unigram: Dict[str, str] = {}

U_PROB = 0.3
T_PROB = 0.3
B_PROB = 0.4
PROB_BIGRAM_B = 0.5
PROB_UNIGRAM_A2 = 0.5
PROB_NULL_Y = 0.1

def respace_text(text: str, u: float, b: float, t: float) -> List[str]:
    filtered = ''.join(ch for ch in text.lower() if 'a' <= ch <= 'z')
    idx = 0
    tokens: List[str] = []
    n_total = len(filtered)
    while idx < n_total:
        r = random.random()
        if r < u:
            n = 1
        elif r < u + b:
            n = 2
        else:
            n = 3
        tokens.append(filtered[idx: idx + n])
        idx += n
    return [tok for tok in tokens if tok]

def encrypt_token(token: str) -> str:
    L = len(token)
    if L == 1:
        use_a2 = (random.random() <= PROB_UNIGRAM_A2)
        table = alphabet2 if use_a2 else alphabet3_unigram
        glyph = table[token.upper()]
    elif L == 2:
        use_B = (random.random() <= PROB_BIGRAM_B)  # B: A1+A3  |  A: A2+A3
        first = (alphabet1 if use_B else alphabet2)[token[0].upper()]
        second = alphabet3[token[1].upper()]
        glyph = first + second
    elif L == 3:
        glyph = (
            alphabet1[token[0].upper()] +
            alphabet2[token[1].upper()] +
            alphabet3[token[2].upper()]
        )
    else:
        raise ValueError("Voynichesque supports unigram, bigram, trigram tokens only.")
    if random.random() < PROB_NULL_Y:
        glyph += 'y'
    return glyph

def voynichesque_encrypt(text: str) -> str:
    tokens = respace_text(text, U_PROB, B_PROB, T_PROB)
    return ' '.join(encrypt_token(tok) for tok in tokens)


# ========= PARAMETER SWEEP =========
def sample_glyph_len_ratio() -> List[int]:
    while True:
        r4 = random.randint(0, 2)
        rem = 20 - r4
        r1_max = min(10, rem)
        r1 = random.randint(0, r1_max)
        rem2 = rem - r1
        r2_max = min(13, rem2)
        r2 = random.randint(0, r2_max)
        r3 = rem2 - r2
        if 0 <= r3 <= 13:
            return [r1, r2, r3, r4]

def sample_respace_probs() -> Tuple[float, float, float]:
    u = random.random() * (0.98)
    t = random.random() * (1.0 - u)
    b = 1.0 - (u + t)
    return u, t, b

def sample_other_probs() -> Tuple[float, float, float]:
    return random.random(), random.random(), random.random()

@dataclass
class _ParamsWrap:
    params: VoynParams

def sample_params() -> VoynParams:
    r = sample_glyph_len_ratio()
    u, t, b = sample_respace_probs()
    pB, pA2, pY = sample_other_probs()
    return VoynParams(
        glyph_len_ratio=r,
        u_prob=u, t_prob=t, b_prob=b,
        prob_bigram_B=pB,
        prob_unigram_A2=pA2,
        prob_null_y=pY
    )

def run_voynichesque_once(
    plaintext: str,
    options: List[Option],
    rng_seed: Optional[int] = None,
    params: Optional[VoynParams] = None
) -> Tuple[str, Tuple[Dict[str, str], Dict[str, str], Dict[str, str]], Dict[str, Any]]:
    if rng_seed is not None:
        random.seed(rng_seed)
    if params is None:
        params = sample_params()

    global U_PROB, T_PROB, B_PROB, PROB_BIGRAM_B, PROB_UNIGRAM_A2, PROB_NULL_Y
    U_PROB = params.u_prob
    T_PROB = params.t_prob
    B_PROB = params.b_prob
    PROB_BIGRAM_B   = params.prob_bigram_B
    PROB_UNIGRAM_A2 = params.prob_unigram_A2
    PROB_NULL_Y     = params.prob_null_y

    ratio = params.glyph_len_ratio
    a1 = build_cipher_alphabet(options, alphabet_id=1, length_ratio=ratio)
    a2 = build_cipher_alphabet(options, alphabet_id=2, length_ratio=ratio)
    a3_base = build_cipher_alphabet(options, alphabet_id=3, length_ratio=ratio)

    global alphabet1, alphabet2, alphabet3, alphabet3_base, alphabet3_unigram
    alphabet1 = a1
    alphabet2 = a2
    alphabet3_base = a3_base
    alphabet3 = alphabet3_base
    alphabet3_unigram = {k: 'd' + v for k, v in alphabet3_base.items()}

    ciphertext = voynichesque_encrypt(plaintext)
    return ciphertext, (a1, a2, a3_base), asdict(params)

def sweep_voynichesque(
    plaintext: str,
    options: List[Option],
    n_runs: int = 20,
    rng_seed: Optional[int] = None,
    keep_alphabets: bool = False
) -> List[Dict[str, Any]]:
    if rng_seed is not None:
        random.seed(rng_seed)
    results = []
    for _ in range(n_runs):
        try:
            params = sample_params()
            ct, alphs, p = run_voynichesque_once(
                plaintext=plaintext,
                options=options,
                rng_seed=None,
                params=params
            )
            results.append({
                "params": p,
                "ciphertext": ct,
                "alphabets": alphs if keep_alphabets else None,
                "error": None
            })
        except Exception as e:
            results.append({
                "params": asdict(params) if 'params' in locals() else None,
                "ciphertext": None,
                "alphabets": None,
                "error": str(e)
            })
    return results


# ========= GLYPH-AWARE METRICS =========
def parse_into_glyphs(text: str) -> List[str]:
    glyphs: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        for cluster in GLYPH_CLUSTERS:
            if text.startswith(cluster, i):
                glyphs.append(cluster)
                i += len(cluster)
                break
        else:
            glyphs.append(text[i])
            i += 1
    return glyphs

def token_lengths(tokens: List[str]) -> List[int]:
    return [len(parse_into_glyphs(tok)) for tok in tokens]

def pct_len_hist(lengths: List[int], max_len: int = 12) -> Dict[str, float]:
    if not lengths:
        return {f"len{L}_pct": 0.0 for L in range(1, max_len+1)}
    capped = [L if L <= max_len else max_len for L in lengths]
    counts = Counter(capped)
    total = len(capped)
    return {f"len{L}_pct": counts.get(L, 0) / total for L in range(1, max_len+1)}

def shannon_entropy_glyph(ciphertext: str) -> float:
    all_glyphs: List[str] = []
    for tok in ciphertext.split():
        all_glyphs.extend(parse_into_glyphs(tok))
    if not all_glyphs:
        return 0.0
    counts = Counter(all_glyphs)
    n = len(all_glyphs)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def conditional_entropy_glyph(ciphertext: str) -> float:
    all_glyphs: List[str] = []
    for tok in ciphertext.split():
        all_glyphs.extend(parse_into_glyphs(tok))
    if len(all_glyphs) < 2:
        return 0.0
    prev_counts = Counter()
    trans_counts = Counter()
    for i in range(len(all_glyphs) - 1):
        a, b = all_glyphs[i], all_glyphs[i+1]
        prev_counts[a] += 1
        trans_counts[(a, b)] += 1
    n_prev = sum(prev_counts.values())
    H = 0.0
    for a, na in prev_counts.items():
        pa = na / n_prev
        nexts = {b: c for (aa, b), c in trans_counts.items() if aa == a}
        Ha = 0.0
        for b, cab in nexts.items():
            p_b_given_a = cab / na
            Ha += -p_b_given_a * math.log2(p_b_given_a)
        H += pa * Ha
    return H

def summarize_ciphertext(ciphertext: str) -> Dict[str, Any]:
    tokens = [t for t in ciphertext.split() if t]
    types = sorted(set(tokens))
    tlens = token_lengths(tokens)
    typelens = token_lengths(types)

    n_tokens = len(tokens)
    n_types = len(types)
    ttr = (n_types / n_tokens) if n_tokens else 0.0
    mean_len = (sum(tlens) / n_tokens) if n_tokens else 0.0
    if n_tokens:
        sl = sorted(tlens)
        median_len = sl[n_tokens // 2] if n_tokens % 2 == 1 else (sl[n_tokens // 2 - 1] + sl[n_tokens // 2]) / 2
    else:
        median_len = 0.0

    H = shannon_entropy_glyph(ciphertext)
    Hc = conditional_entropy_glyph(ciphertext)

    token_hist = pct_len_hist(tlens, max_len=12)
    type_hist  = pct_len_hist(typelens, max_len=12)
    type_hist  = {f"type_len{L}_pct": type_hist[f"len{L}_pct"] for L in range(1, 13)}

    return {
        "n_tokens": n_tokens,
        "n_types": n_types,
        "ttr": ttr,
        "mean_token_len": mean_len,
        "median_token_len": median_len,
        "char_entropy": H,
        "cond_char_entropy": Hc,
        **token_hist,
        **type_hist
    }


# ========= ALPHABET CODE (compact, recoverable) =========
def _alph_to_list(alpha: Dict[str, str]) -> List[str]:
    # A..Z order
    return [alpha[ch] for ch in string.ascii_uppercase]

def build_alphabet_code(a1: Dict[str, str], a2: Dict[str, str], a3: Dict[str, str]) -> str:
    payload = {"v": 1, "A1": _alph_to_list(a1), "A2": _alph_to_list(a2), "A3": _alph_to_list(a3)}
    s = json.dumps(payload, separators=(',', ':'))
    code = base64.urlsafe_b64encode(s.encode("utf-8")).decode("utf-8").rstrip('=')
    return code


# ========= CSV LOGGING =========
def init_csv_logger(path: str) -> List[str]:
    fieldnames = [
        # params
        "r1","r2","r3","r4",
        "u_prob","b_prob","t_prob",
        "prob_bigram_B","prob_unigram_A2","prob_null_y",
        # encoded alphabets
        "alphabet_code",
        # summary stats
        "n_tokens","n_types","ttr","mean_token_len","median_token_len",
        "char_entropy","cond_char_entropy",
    ] + [f"len{L}_pct" for L in range(1,13)] + [f"type_len{L}_pct" for L in range(1,13)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return fieldnames

def append_run_to_csv(path: str, params_dict: Dict[str, Any], summary_dict: Dict[str, Any], alphabet_code: str) -> None:
    row = {
        # params
        "r1": params_dict["glyph_len_ratio"][0],
        "r2": params_dict["glyph_len_ratio"][1],
        "r3": params_dict["glyph_len_ratio"][2],
        "r4": params_dict["glyph_len_ratio"][3],
        "u_prob": params_dict["u_prob"],
        "b_prob": params_dict["b_prob"],
        "t_prob": params_dict["t_prob"],
        "prob_bigram_B": params_dict["prob_bigram_B"],
        "prob_unigram_A2": params_dict["prob_unigram_A2"],
        "prob_null_y": params_dict["prob_null_y"],
        # alphabets
        "alphabet_code": alphabet_code,
        # metrics
        **summary_dict
    }
    fieldnames = [
        "r1","r2","r3","r4",
        "u_prob","b_prob","t_prob",
        "prob_bigram_B","prob_unigram_A2","prob_null_y",
        "alphabet_code",
        "n_tokens","n_types","ttr","mean_token_len","median_token_len",
        "char_entropy","cond_char_entropy",
    ] + [f"len{L}_pct" for L in range(1,13)] + [f"type_len{L}_pct" for L in range(1,13)]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


# ========= MAIN =========
if __name__ == "__main__":
    SWEEP_SEED = 123
    random.seed(SWEEP_SEED)

    all_options = load_options(CSV_PATH)
    init_csv_logger(LOG_CSV_PATH)

        # === Load reference plaintext from TXT file ===
    TXT_FILE_PATH = "input/examples/nathist_book16.txt"
    
    with open(TXT_FILE_PATH, "r", encoding="utf-8") as f:
        plaintext = f.read()
    
    # Normalize whitespace:
    plaintext = " ".join(plaintext.split())
    
    # Then run the sweep using that plaintext
    runs = sweep_voynichesque(
        plaintext=plaintext,
        options=all_options,
        n_runs=10, # Experimentally, this script is ~75.5% efficient. That is, set n_runs=1000, you'll get ~755 successful runs.
        rng_seed=SWEEP_SEED,
        keep_alphabets=True # we need alphabets to build the code
    )

    success = 0
    for r in runs:
        if r["error"]:
            continue
        params = r["params"]
        ct = r["ciphertext"]
        (A1, A2, A3) = r["alphabets"]
        summary = summarize_ciphertext(ct)
        code = build_alphabet_code(A1, A2, A3)
        append_run_to_csv(LOG_CSV_PATH, params, summary, code)
        success += 1

    print(f"Completed {len(runs)} runs; {success} successful.")
    print(f"Sweep log saved to: {LOG_CSV_PATH}")
