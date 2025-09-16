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


import random
import re
import unicodedata
from collections import defaultdict
import pandas as pd

# === Parameters ===
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")
TABLES = ['alpha', 'beta1', 'beta2', 'beta3', 'gamma1', 'gamma2']
STATES = ['unigram', 'prefix', 'suffix']
RESPACING = 17 # Use 18 for simplified respacing (50-50 unigram-bigram) and 17 for standard respacing (slight bigram excess)
USE_78_CARD_DECK = False  # Toggle between 52-card and 78-card deck
SPACE_REMOVAL_RATE = 0.03      # Fraction of spaces to randomly remove in "respaced" ciphertext output, to mimic uncertain VMS spacing
UNAMBIGUOUS=True #True means bigram token generation avoids accidentally creating unigram word types. Strongly recommend True

# === Card weights ===
CARD_WEIGHTS = {
    False: {'alpha': 20, 'beta1': 8, 'beta2': 8, 'beta3': 8, 'gamma1': 4, 'gamma2': 4},
    True:  {'alpha': 28, 'beta1': 14, 'beta2': 11, 'beta3': 11, 'gamma1': 7, 'gamma2': 7},
}

def generate_placeholder_tables():
    table = defaultdict(dict)
    for table_name in TABLES:
        for state in STATES:
            for letter in ALPHABET:
                key = f"{state}_{table_name}_{letter}"
                table[table_name][(state, letter)] = key
    return table

naibbe_tables = generate_placeholder_tables()

# === Load glyph mappings from CSV ===
glyph_df = pd.read_csv("references/naibbe_tables.csv")  # Ensure this file is present
placeholder_to_glyph = dict(zip(glyph_df['code'], glyph_df['glyphs']))

# Precompute all unigram glyphs for ambiguity checking
unigram_glyphs = {
    glyph for code, glyph in placeholder_to_glyph.items()
    if code.startswith("unigram_")
}


# === Deck creation ===
def create_card_deck(use_78=False):
    weights = CARD_WEIGHTS[use_78]
    deck = []
    for table, count in weights.items():
        deck.extend([table] * count)
    random.shuffle(deck)
    return deck

# === Respacing ===
def respace_plaintext(text, pre_plaintext_file=None):
    text = text.lower().replace(" ", "")
    i = 0
    output = []
    while i < len(text):
        if i == len(text) - 1 or random.random() < (RESPACING/36):
            output.append(text[i])
            i += 1
        else:
            output.append(text[i:i+2])
            i += 2
    # Write pre-encryption respaced plaintext to file if provided
    if pre_plaintext_file is not None:
        pre_plaintext_file.write(" ".join(output) + "\n")
    return output

# === Ambiguity log ===
ambiguity_retries = 0

# === Encryption ===
def encrypt_naibbe(plaintext, tables, glyph_map, use_78=False, pre_plaintext_file=None):
    ngrams = respace_plaintext(plaintext, pre_plaintext_file)
    ciphertext = []
    deck = create_card_deck(use_78)
    deck_index = 0

    for token in ngrams:
        if len(token) == 1:
            state = 'unigram'
            letters = [token]
        else:
            state = 'bigram'
            letters = [token[0], token[1]]

        if state == 'unigram':
            # === Unigram handling ===
            if deck_index >= len(deck):
                deck = create_card_deck(use_78)
                deck_index = 0
            table = deck[deck_index]
            deck_index += 1
            code = tables[table][('unigram', letters[0])]
            glyph = glyph_map.get(code, code)
            ciphertext.append(glyph)

        else:  # state == 'bigram'
            if UNAMBIGUOUS:
                # === Ambiguity-safe bigram handling ===
                while True:
                    # Prefix
                    if deck_index >= len(deck):
                        deck = create_card_deck(use_78)
                        deck_index = 0
                    table_prefix = deck[deck_index]
                    deck_index += 1
                    code_prefix = tables[table_prefix][('prefix', letters[0])]
                    glyph_prefix = glyph_map.get(code_prefix, code_prefix)

                    # Suffix
                    if deck_index >= len(deck):
                        deck = create_card_deck(use_78)
                        deck_index = 0
                    table_suffix = deck[deck_index]
                    deck_index += 1
                    code_suffix = tables[table_suffix][('suffix', letters[1])]
                    glyph_suffix = glyph_map.get(code_suffix, code_suffix)

                    combined = glyph_prefix + glyph_suffix
                    if combined not in unigram_glyphs:
                        ciphertext.append(combined)
                        break
                    else:
                        global ambiguity_retries
                        ambiguity_retries += 1
            else:
                # === Standard bigram handling ===
                for i, letter in enumerate(letters):
                    if deck_index >= len(deck):
                        deck = create_card_deck(use_78)
                        deck_index = 0
                    table = deck[deck_index]
                    deck_index += 1
                    substate = 'prefix' if i == 0 else 'suffix'
                    code = tables[table][(substate, letter)]
                    glyph = glyph_map.get(code, code)
                    if i == 0:
                        ciphertext.append(glyph)
                    else:
                        ciphertext[-1] += glyph

    return ciphertext

# === Full normalization of line with Latin replacement ===
def clean_line(text):
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

# === Randomly remove spaces in output ===
def respace_line(line, drop_rate):
    if drop_rate <= 0:
        return line.strip()
    if drop_rate >= 1:
        return line.replace(" ", "")

    tokens = line.strip().split()
    if len(tokens) < 2:
        return line.strip()

    output = tokens[0]
    for tok in tokens[1:]:
        if random.random() < drop_rate:
            output += tok
        else:
            output += ' ' + tok
    return output

# === Line-by-line file encryption and optional respacing ===
if __name__ == "__main__":
    input_path = "input/examples/divina_commedia.txt"
    output_path = "encrypted/divcom_output_ciphertext.txt"
    respaced_output_path = "encrypted/divcom_output_ciphertext_respaced.txt"
    pre_plaintext_output_path = "respaced_plaintext/divcom_pre_encryption_respaced_plaintext.txt"

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(respaced_output_path, "w", encoding="utf-8") as frespace, \
         open(pre_plaintext_output_path, "w", encoding="utf-8") as fplain:

        for line in fin:
            cleaned = clean_line(line)
            if cleaned:
                encrypted_tokens = encrypt_naibbe(
                    cleaned, naibbe_tables, placeholder_to_glyph,
                    use_78=USE_78_CARD_DECK,
                    pre_plaintext_file=fplain
                )
                line_out = " ".join(encrypted_tokens)
                fout.write(line_out + "\n")
                frespace.write(respace_line(line_out, SPACE_REMOVAL_RATE) + "\n")
            else:
                fout.write("\n")
                frespace.write("\n")
                fplain.write("\n")

# === print ambiguity count ===
if UNAMBIGUOUS:
    print(f"Total ambiguity retries due to prefix+suffix collisions with unigrams: {ambiguity_retries}")
