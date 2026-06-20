"""Przygotowanie korpusu ABC z thesession.org tunes.csv.
Filtr: jigi 6/8 (spójny styl). Buduje grające bloki ABC + normalizuje tonację.
"""
import csv, re, sys
csv.field_size_limit(10**7)

MODE_TABLE = {
    "major": "", "ionian": "", "minor": "min", "aeolian": "min",
    "dorian": "dor", "mixolydian": "mix", "phrygian": "phr",
    "lydian": "lyd", "locrian": "loc", "": "",
}

def norm_key(mode: str) -> str:
    m = re.match(r"^([A-Ga-g][#b]?)(.*)$", mode.strip())
    if not m:
        return "C"
    root, word = m.group(1), m.group(2).lower()
    return root + MODE_TABLE.get(word, "")

ALLOWED = set("ABCDEFGabcdefg0123456789|:[]()<>/'^_=.,~- zZxX")

def clean_abc(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    body = re.sub(r'"[^"]*"', "", body)       # usuń symbole akordów / adnotacje "..."
    body = re.sub(r"[ \t]+", " ", body)        # scal podwójne spacje po usunięciu
    body = re.sub(r"\n+", "\n", body)
    return body

def main():
    rows_out, n_total, n_kept = [], 0, 0
    with open("data/tunes.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_total += 1
            if "jig" not in row["type"].lower():
                continue
            if row["meter"].strip() != "6/8":
                continue
            body = clean_abc(row["abc"])
            if not (40 <= len(body) <= 500):
                continue
            if any(ch not in ALLOWED for ch in body.replace("\n", "")):
                continue
            key = norm_key(row["mode"])
            block = f"X:1\nM:6/8\nK:{key}\n{body}\n"
            rows_out.append(block)
            n_kept += 1

    text = "\n".join(rows_out)
    with open("data/jigs.abc", "w", encoding="utf-8") as f:
        f.write(text)

    vocab = sorted(set(text))
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"melodii w pliku   : {n_total}")
    print(f"jigi 6/8 zachowane: {n_kept}")
    print(f"znaki łącznie     : {len(text):,}")
    print(f"rozmiar słownika  : {len(vocab)} znaków")
    print(f"słownik           : {''.join(vocab)!r}")
    print("\n--- pierwszy blok ---")
    print(rows_out[0])

if __name__ == "__main__":
    main()
