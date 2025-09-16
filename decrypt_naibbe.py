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

import pandas as pd

# === CONFIG ===
BASIC = True          # If True, prefer unigram decoding when available. Strongly recommend keeping this on True
MARK_COMPOUND = True   # If True, try to resolve compound tokens. Strongly recommend keeping this on True

# --- small helper to deduplicate while preserving order ---
def _uniq(seq):
    return list(dict.fromkeys(seq))

# === Step 1: Build reverse mappings ===
def build_reverse_mappings(glyph_df):
    unigram_map = {}
    prefix_map = {}
    suffix_map = {}
    all_word_types = set()

    for _, row in glyph_df.iterrows():
        code = row['code']
        glyph = row['glyphs']
        all_word_types.add(glyph)
        if code.startswith("unigram_"):
            letter = code.split('_')[-1]
            unigram_map[glyph] = letter
        elif code.startswith("prefix_"):
            letter = code.split('_')[-1]
            prefix_map[glyph] = letter
        elif code.startswith("suffix_"):
            letter = code.split('_')[-1]
            suffix_map[glyph] = letter
    return unigram_map, prefix_map, suffix_map, all_word_types

# === Step 2: Decrypt one token ===
def decrypt_naibbe_token(token, unigram_map, prefix_map, suffix_map, all_word_types, basic=True, compound=True):
    unigram = unigram_map.get(token)
    bigram_options = []

    # Try all prefix/suffix splits
    for i in range(1, len(token)):
        pre = token[:i]
        suf = token[i:]
        if pre in prefix_map and suf in suffix_map:
            letter1 = prefix_map[pre]
            letter2 = suffix_map[suf]
            bigram_options.append(letter1 + letter2)

    bigram_options = _uniq(bigram_options)  # de-dupe, keep order

    if basic:
        # BASIC mode still prefers unigram if available
        if unigram:
            return unigram
        # If token parses as a bigram, show ALL valid bigram readings if ambiguous
        if bigram_options:
            if len(bigram_options) == 1:
                return bigram_options[0]
            else:
                return "(" + "|".join(bigram_options) + ")"
        # Otherwise, optionally try compound
        if MARK_COMPOUND and compound:
            return decrypt_compound_token(token, unigram_map, prefix_map, suffix_map, all_word_types, basic)
        return "[?]"
    else:
        possibilities = []
        if unigram:
            possibilities.append(unigram)
        possibilities.extend(bigram_options)

        if not possibilities:
            if MARK_COMPOUND and compound:
                return decrypt_compound_token(token, unigram_map, prefix_map, suffix_map, all_word_types, basic)
            else:
                return "[?]"
        elif len(possibilities) == 1:
            return possibilities[0]
        else:
            return "(" + "|".join(_uniq(possibilities)) + ")"

# === Step 3: Decrypt compound token ===
def decrypt_compound_token(token, unigram_map, prefix_map, suffix_map, all_word_types, basic=True):
    N = len(token)
    candidates = []

    for i in range(1, N):
        left = token[:i]
        right = token[i:]

        # EXACTLY two tokens: we disable compound recursion but still allow bigram ambiguity display
        left_decoded = decrypt_naibbe_token(
            left, unigram_map, prefix_map, suffix_map, all_word_types, basic=basic, compound=False
        ) if left in all_word_types else "[?]"

        right_decoded = decrypt_naibbe_token(
            right, unigram_map, prefix_map, suffix_map, all_word_types, basic=basic, compound=False
        ) if right in all_word_types else "[?]"

        if "[?]" not in (left_decoded, right_decoded):
            candidates.append((left_decoded + " " + right_decoded, abs(len(left) - len(right))))

    if not candidates:
        return "[?]"

    # Mark the closest to even split
    candidates.sort(key=lambda x: x[1])
    best_candidate = candidates[0][0] + "*"
    others = [c[0] for c in candidates[1:]]

    return best_candidate if not others else "(" + "|".join([best_candidate] + others) + ")"

# === Step 4: Decrypt a full line ===
def decrypt_naibbe_line(line, unigram_map, prefix_map, suffix_map, all_word_types, basic=True):
    tokens = line.strip().split()
    return " ".join(
        decrypt_naibbe_token(token, unigram_map, prefix_map, suffix_map, all_word_types, basic)
        for token in tokens
    )

# === Step 5: Read file, decrypt, write output ===
def decrypt_naibbe_file(input_path="encrypted/divcom_output_ciphertext.txt",
                        output_path="decrypted/divcom_output_ciphertext_decrypted.txt",
                        glyph_table_path="references/naibbe_tables.csv",
                        basic=True):
    glyph_df = pd.read_csv(glyph_table_path)
    unigram_map, prefix_map, suffix_map, all_word_types = build_reverse_mappings(glyph_df)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            decrypted = decrypt_naibbe_line(line, unigram_map, prefix_map, suffix_map, all_word_types, basic)
            fout.write(decrypted + "\n")

# === ENTRY POINT ===
if __name__ == "__main__":
    decrypt_naibbe_file(basic=BASIC)
