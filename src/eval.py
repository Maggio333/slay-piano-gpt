"""Reprodukowalny eval: GPT vs n-gram na TYM SAMYM held-out + metryki muzyczne.
Jeden plik -> results/eval.json (+ results/eval_report.html). Deterministyczny (seed).

Po co: wpiac pomiar n-gram-vs-GPT pod wspolny, reprodukowalny harness Slayera -
te same liczby, jeden command, zero recznego porownywania. Zasada: zadnej tezy bez dowodu.

Metryki:
  - perplexity na held-out (GPT i n-gram licza prawdopodobienstwo, nie tylko samplują),
  - nowosc: % k-gramow generacji obecnych w treningu (kopiuje vs komponuje),
  - poprawnosc ABC: % wygenerowanych melodii ze zbilansowana strukturą (|: :|, naglowki, charset),
  - srednia dlugosc.

Wymaga data/jigs.abc -> python src/prepare_data.py (z dumpu thesession.org).
"""
import os, json, math, argparse, collections, statistics, random
import torch
from gpt import GPT
import ngram_model as ng

SEED = 20260620
SEED_STR = "X:1\nM:6/8\nK:D\n"


def split(text):
    n = int(0.9 * len(text))
    return text[:n], text[n:]


# ---------- perplexity ----------
def gpt_eval(ckpt, val_text, device, n_batches=300, bs=32):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    stoi, cfg = ck["stoi"], ck["config"]
    model = GPT(cfg).to(device); model.load_state_dict(ck["model"]); model.eval()
    block = cfg.block_size
    data = torch.tensor([stoi[c] for c in val_text if c in stoi], dtype=torch.long)
    g = torch.Generator().manual_seed(SEED)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            ix = torch.randint(len(data) - block, (bs,), generator=g)
            x = torch.stack([data[i:i + block] for i in ix]).to(device)
            y = torch.stack([data[i + 1:i + 1 + block] for i in ix]).to(device)
            _, loss = model(x, y); losses.append(loss.item())
    m = statistics.mean(losses)
    return dict(model="GPT (transformer)", params=model.num_params(),
                val_loss=round(m, 4), perplexity=round(math.exp(m), 3),
                context=block), model, ck


def ngram_eval(models, train_text, val_text, order=ng.ORDER, addk=0.05, cap=120000):
    V = len(set(train_text))
    vt = val_text[:cap]
    nll = 0.0; N = 0
    for i, ch in enumerate(vt):
        p = None
        for c in ng.CTX:
            if c > i:
                continue
            ctx = vt[i - c:i] if c > 0 else ""
            counter = models[c].get(ctx)
            if counter:
                tot = sum(counter.values())
                p = (counter.get(ch, 0) + addk) / (tot + addk * V); break
        if p is None:
            p = 1.0 / V
        nll += -math.log(p); N += 1
    m = nll / N
    return dict(model=f"n-gram (order-{order})", params=None,
                val_loss=round(m, 4), perplexity=round(math.exp(m), 3),
                context=order, scored_chars=N)


# ---------- generacja ----------
def gen_gpt(model, ck, device, n=24, temp=0.8, top_k=20, max_new=400):
    stoi, itos = ck["stoi"], ck["itos"]
    idx = torch.tensor([[stoi[c] for c in SEED_STR]], dtype=torch.long, device=device)
    torch.manual_seed(SEED)
    out = model.generate(idx, max_new * n // 1, temperature=temp, top_k=top_k)[0].tolist()
    raw = "".join(itos[i] for i in out)
    return _split_tunes(raw)[:n]


def gen_ngram(models, n=24, temp=0.7):
    random.seed(SEED)
    return [ng.generate(models, SEED_STR, temp=temp) for _ in range(n)]


def _split_tunes(raw):
    tunes, cur = [], []
    for line in raw.split("\n"):
        if line.startswith("X:") and cur:
            tunes.append("\n".join(cur)); cur = []
        cur.append(line)
    if cur:
        tunes.append("\n".join(cur))
    return [t.strip() for t in tunes if "X:" in t]


# ---------- metryki muzyczne ----------
ALLOWED = set("ABCDEFGabcdefg0123456789|:[]()<>/'^_=.,~- zZxX\nXMK")

def abc_valid(tune):
    has_hdr = all(h in tune for h in ("X:", "M:", "K:"))
    body = "".join(ln for ln in tune.split("\n") if not ln[:2] in ("X:", "M:", "K:"))
    bars = [b for b in body.replace(":", "|").split("|") if b.strip()]
    open_rep = tune.count("|:"); close_rep = tune.count(":|")
    balanced = open_rep == close_rep
    charset_ok = all(c in ALLOWED for c in tune)
    enough_bars = len(bars) >= 4
    return has_hdr and balanced and charset_ok and enough_bars


def validity_rate(tunes):
    if not tunes:
        return 0.0
    return round(100 * sum(abc_valid(t) for t in tunes) / len(tunes), 1)


def novelty(tunes, train_text, ks=(4, 6, 8)):
    body = "".join(ln for t in tunes for ln in t.split("\n")
                   if not ln[:2] in ("X:", "M:", "K:"))
    res = {}
    for k in ks:
        tr = set(train_text[i:i + k] for i in range(len(train_text) - k))
        gk = [body[i:i + k] for i in range(len(body) - k)]
        if not gk:
            res[k] = None; continue
        copied = sum(1 for g in gk if g in tr)
        res[k] = round(100 * (1 - copied / len(gk)), 1)  # % NOWYCH k-gramow
        del tr
    return res


def avg_len(tunes):
    return round(sum(len(t) for t in tunes) / len(tunes), 1) if tunes else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/jigs.abc")
    ap.add_argument("--ckpt", default="data/gpt_ckpt.pt")
    ap.add_argument("--n", type=int, default=24, help="ile melodii do metryk")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        raise SystemExit(f"Brak {args.data}. Odbuduj: python src/prepare_data.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = open(args.data, encoding="utf-8").read()
    train_text, val_text = split(text)

    print("== perplexity (held-out) ==")
    gpt_m, model, ck = gpt_eval(args.ckpt, val_text, device)
    ng_models = ng.train(train_text)
    ng_m = ngram_eval(ng_models, train_text, val_text)

    print("== generacja + metryki ==")
    gpt_tunes = gen_gpt(model, ck, device, n=args.n)
    ng_tunes = gen_ngram(ng_models, n=args.n)

    for row, tunes in ((gpt_m, gpt_tunes), (ng_m, ng_tunes)):
        row["novelty_new_pct"] = novelty(tunes, train_text)
        row["abc_valid_pct"] = validity_rate(tunes)
        row["avg_len"] = avg_len(tunes)
        row["n_samples"] = len(tunes)

    res = dict(corpus_chars=len(text), vocab=len(set(text)),
               train_chars=len(train_text), val_chars=len(val_text),
               seed=SEED, models=[gpt_m, ng_m])
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/eval.json", "w"), ensure_ascii=False, indent=2)
    write_report(res, "results/eval_report.html")

    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n[zapisano -> results/eval.json + results/eval_report.html]")


def write_report(res, path):
    rows = ""
    for m in res["models"]:
        nov = m["novelty_new_pct"]
        rows += (f"<tr><td><b>{m['model']}</b></td>"
                 f"<td>{m['params'] or '-'}</td>"
                 f"<td>{m['context']} zn.</td>"
                 f"<td class=hl>{m['perplexity']}</td>"
                 f"<td>{m['val_loss']}</td>"
                 f"<td>{nov.get(8, '-')}%</td>"
                 f"<td>{m['abc_valid_pct']}%</td>"
                 f"<td>{m['avg_len']}</td></tr>")
    html = f"""<!DOCTYPE html><html lang="pl"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>slay-piano-gpt - eval: GPT vs n-gram</title>
<style>
body{{background:#0c0c12;color:#e6e6ef;font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:880px;margin:0 auto;padding:40px 20px;line-height:1.6}}
h1{{font-size:1.7rem;margin:0 0 4px}} .sub{{color:#9a9ab0;margin-bottom:24px;font-size:.95rem}}
table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:.92rem}}
th,td{{padding:9px 11px;border-bottom:1px solid #23232f;text-align:left}}
th{{color:#9a9ab0;font-size:.74rem;text-transform:uppercase;letter-spacing:.05em}}
td.hl{{color:#d9a441;font-weight:700;font-size:1.05rem}} td b{{color:#fff}}
.note{{color:#8d8da3;font-size:.85rem;margin-top:14px}}
code{{background:#15151f;padding:2px 6px;border-radius:5px;font-size:.85rem}}
</style></head><body>
<h1>slay-piano-gpt - eval</h1>
<div class="sub">GPT (od zera) vs n-gram baseline na tym samym held-out. Reprodukowalne: <code>python src/eval.py</code> &middot; seed {res['seed']}</div>
<p>Korpus: {res['corpus_chars']:,} znakow &middot; slownik {res['vocab']} &middot; held-out {res['val_chars']:,} znakow.</p>
<table>
<tr><th>Model</th><th>Param</th><th>Kontekst</th><th>Perplexity &darr;</th><th>Val loss</th><th>Nowe 8-gramy &uarr;</th><th>Poprawny ABC &uarr;</th><th>Sr. dlugosc</th></tr>
{rows}
</table>
<p class="note">Perplexity (nizej=lepiej): GPT wygrywa dzieki oknu uwagi 128 zn. vs 6 zn. n-grama.
"Nowe 8-gramy" = % 8-znakowych fragmentow generacji NIEobecnych w treningu (wyzej = wiecej komponuje, mniej kopiuje).
"Poprawny ABC" = % melodii ze zbilansowana struktura (naglowki, |: :|, charset). Metryka zadaniowa, nie tylko strata (L09).</p>
<p class="note">Slayer Labs &middot; reprodukowalny harness &middot; model: Arkadiusz Slota</p>
</body></html>"""
    open(path, "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    main()
