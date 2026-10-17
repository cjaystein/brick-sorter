#!/usr/bin/env python3
"""
Build the head matcher: a "fingerprint" of every minifigure head picture on Rebrickable, so the web
app can show which heads a photo of a head looks most like.

    python build_heads.py                   # uses rb-index.bin.gz and rb-pics.txt.gz next to this file

It downloads a general image model (DINOv2 small, from Hugging Face, once) to heads-model.onnx, every
head picture (about 5,500: one per head print and colour) from Rebrickable, and writes:

  heads-index.bin.gz   "HDIX" + uint32 header length + header JSON + float16 fingerprints (n x dim)
                       header: built, model, dim, n, prep (how pictures are prepared), heads [[part, colour id]]
  heads-eval.json      how often the right head comes first / in the top 5 for pictures made deliberately
                       worse (blur, rotation, lighting, crop, a tray instead of a white background)

The web app prepares a photo of a head exactly as prep() does here before fingerprinting it.
Only standard Python plus numpy, pillow and onnxruntime.
"""

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import random
import struct
import sys
import time
import urllib.request
from array import array
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter

# Tested 17 Oct 2026 on 600 heads made worse: the small (quantised) model with several views per head did
# best (42% right first, 56% in the top 5); the full model (88 MB) did no better. Brickognize did better still
# on the same pictures (76% / 79%), so the app uses this only as an opt-in second opinion.
MODELS = {"heads-model.onnx": "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx"}
BRICKOGNIZE = "https://api.brickognize.com/predict/parts/?top_k_items=10"
PICS = "https://cdn.rebrickable.com/media/parts/"
SIZE, MARGIN = 224, 0.08
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
UA = {"User-Agent": "BrickSorter-heads-builder/1.0"}


def load_index(path):
    raw = gzip.open(path).read()
    n = struct.unpack("<I", raw[4:8])[0]
    return json.loads(raw[8:8 + n])


def head_pictures(root):
    h = load_index(root / "rb-index.bin.gz")
    cat = next(int(k) for k, v in h["categories"].items() if v == "Minifig Heads")
    heads = {p.lower() for p, c in zip(h["parts"], h["part_cats"]) if c == cat}
    out = []
    for line in gzip.open(root / "rb-pics.txt.gz", "rt"):
        a = line.rstrip("\n").split("\t")
        if len(a) == 3 and a[0].lower() in heads:
            out.append((a[0], int(a[1]), a[2]))
    return sorted(out)


def fetch(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(2 * (t + 1))


def prep(img):
    """The picture as the model sees it: on white, cropped to the part plus a margin, square, 224 px.
    (The app removes the tray from a photo first, so both sides look alike.)"""
    img = img.convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("RGB")
    a = np.asarray(img).astype(np.int16)
    fg = np.abs(a - 255).sum(-1) > 40
    ys, xs = np.nonzero(fg)
    if len(xs):
        img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = int(max(img.size) * (1 + 2 * MARGIN)) or 1
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    sq = sq.resize((SIZE, SIZE), Image.BILINEAR)
    x = (np.asarray(sq).astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class Model:
    """Two fingerprints from one run: "cls" (the model's summary of the picture) and "cls+mean"
    (that plus the average of its 256 detail patches, each made unit length, then together)."""
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        names = [o.name for o in self.s.get_outputs()]
        self.out = "last_hidden_state" if "last_hidden_state" in names else names[0]

    def __call__(self, x):
        y = self.s.run([self.out], {self.inp: x})[0][0]   # tokens x dim
        cls, mean = unit(y[0]), unit(y[1:].mean(0))
        return {"cls": cls, "cls+mean": unit(np.concatenate([cls, mean]))}


def views(img, rng):
    """Slightly worse copies of a reference picture, so its fingerprint tolerates a so-so photo."""
    img = img.convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("RGB")
    w, h = img.size
    return [img,
            img.filter(ImageFilter.GaussianBlur(1.2)),
            img.resize((max(24, w // 3), max(24, h // 3)), Image.BILINEAR),
            img.rotate(12, resample=Image.BICUBIC, fillcolor=(255, 255, 255)),
            img.rotate(-12, resample=Image.BICUBIC, fillcolor=(255, 255, 255))]


def brickognize(img):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=90)
    bd = "----bricksorter"
    body = (f"--{bd}\r\nContent-Disposition: form-data; name=\"query_image\"; filename=\"q.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + b.getvalue() + f"\r\n--{bd}--\r\n".encode()
    for t in range(4):
        try:
            req = urllib.request.Request(BRICKOGNIZE, data=body, headers={"Content-Type": f"multipart/form-data; boundary={bd}", **UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return [it["id"] for it in json.load(r).get("items", [])]
        except Exception:
            time.sleep(3 * (t + 1))
    return None


def worse(img, rng):
    """A head picture made to look like a so-so phone photo of a head on a tray."""
    img = img.convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("RGB")
    img = img.rotate(rng.uniform(-25, 25), resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    w, h = img.size
    dx, dy = int(w * rng.uniform(-0.06, 0.06)), int(h * rng.uniform(-0.06, 0.06))
    img = img.crop((max(0, dx), max(0, dy), w + min(0, dx), h + min(0, dy)))
    img = img.resize((max(24, int(w * rng.uniform(0.35, 0.8))),) * 2, Image.BILINEAR)   # small in the photo
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.4)))
    a = np.asarray(img).astype(np.float32)
    a = a * np.array([rng.uniform(0.92, 1.06), 1, rng.uniform(0.9, 1.05)]) + np.random.default_rng(rng.randint(0, 9999)).normal(0, 5, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    b = io.BytesIO(); img.save(b, "JPEG", quality=rng.randint(55, 85))
    return Image.open(io.BytesIO(b.getvalue()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--model", default="heads-model.onnx")
    ap.add_argument("--cache", default="/tmp/head-pictures")
    ap.add_argument("--limit", type=int, default=0, help="only the first N heads (for a quick test)")
    ap.add_argument("--eval", type=int, default=600, help="how many heads to test with worse pictures")
    ap.add_argument("--brickognize", type=int, default=200, help="how many of those to also ask Brickognize, for comparison")
    args = ap.parse_args()
    root, cache = Path(args.root), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    heads = head_pictures(root)
    if args.limit:
        heads = heads[:args.limit]
    print(f"{len(heads)} head pictures")

    def get(h):
        f = cache / h[2].replace("/", "_")
        if not f.exists():
            data = fetch(PICS + h[2])
            if not data:
                return h, None
            f.write_bytes(data)
        try:
            return h, Image.open(f).copy()
        except Exception:
            return h, None

    with cf.ThreadPoolExecutor(16) as ex:
        got = [(h, img) for h, img in ex.map(get, heads) if img is not None]
    print(f"{len(got)} pictures ready")
    rng = random.Random(7)
    sample = rng.sample(range(len(got)), min(args.eval, len(got)))
    queries = {i: worse(got[i][1], random.Random(1000 + i)) for i in sample}
    prints = [h[0].lower() for h, _ in got]

    def score(V, Q):
        ranks, pranks = [], []
        for i, q in zip(sample, Q):
            order = np.argsort(-(V @ q))
            ranks.append(int(np.nonzero(order == i)[0][0]) + 1)
            pranks.append(next(k + 1 for k, j in enumerate(order) if prints[j] == prints[i]))
        sh = lambda rs, k: round(sum(r <= k for r in rs) / len(rs), 3)
        return {"top1": sh(ranks, 1), "top5": sh(ranks, 5), "top10": sh(ranks, 10), "print_top5": sh(pranks, 5)}

    results, best = {}, None
    for name, url in MODELS.items():
        path = root / name if args.model == "heads-model.onnx" else Path(args.model)
        if not path.exists():
            print(f"Downloading {name}...")
            data = fetch(url)
            if not data:
                print(f"  couldn't download {name}, skipping"); continue
            path.write_bytes(data)
        model = Model(path)
        t0 = time.time()
        plain = [model(prep(img)) for _, img in got]
        Q = [model(prep(queries[i])) for i in sample]
        per = (time.time() - t0) / (len(got) + len(sample))
        for feat in ["cls", "cls+mean"]:
            V = np.stack([v[feat] for v in plain]).astype(np.float16).astype(np.float32)
            key = f"{name} {feat}"
            results[key] = {**score(V, [q[feat] for q in Q]), "seconds_per_picture": round(per, 3), "mb": round(path.stat().st_size / 1e6, 1)}
            print(key, results[key])
        # several views per reference, for the better fingerprint of this model
        feat = max(["cls", "cls+mean"], key=lambda f: results[f"{name} {f}"]["top5"])
        multi = []
        for k, (_, img) in enumerate(got):
            vs = [model(prep(v))[feat] for v in views(img, rng)]
            multi.append(unit(np.mean(vs, 0)))
        V = np.stack(multi).astype(np.float16).astype(np.float32)
        key = f"{name} {feat} +views"
        results[key] = {**score(V, [q[feat] for q in Q]), "seconds_per_picture": round(per, 3), "mb": round(path.stat().st_size / 1e6, 1)}
        print(key, results[key])
        for key2, V2 in [(f"{name} {feat}", np.stack([v[feat] for v in plain])), (key, V)]:
            r = results[key2]
            # prefer the small model unless the big one is clearly better (it's 4x the download)
            bonus = 0.03 if name == "heads-model.onnx" else 0
            if best is None or r["top5"] + bonus > best[0]:
                best = (r["top5"] + bonus, name, feat, key2.endswith("+views"), V2)
    # Brickognize on the same pictures, for comparison (BrickLink numbers turned into Rebrickable's)
    h0 = load_index(root / "rb-index.bin.gz")
    bl2rb = {}
    for p, ids in (h0.get("bl") or {}).items():
        for b in ids.split(","):
            bl2rb.setdefault(b.lower(), p.lower())
    bg_r = []
    for i in sample[:args.brickognize]:
        ids = brickognize(queries[i])
        if ids is None:
            continue
        rbs = [bl2rb.get(x.lower(), x.lower()) for x in ids]
        bg_r.append(next((k + 1 for k, x in enumerate(rbs) if x == prints[i]), 99))
        time.sleep(0.4)
    if bg_r:
        results["Brickognize (same pictures)"] = {"tested": len(bg_r), "print_top1": round(sum(r <= 1 for r in bg_r) / len(bg_r), 3), "print_top5": round(sum(r <= 5 for r in bg_r) / len(bg_r), 3)}
        print("Brickognize", results["Brickognize (same pictures)"])

    _, name, feat, multi, V = best
    V = unit(V.astype(np.float32))
    header = {"built": time.strftime("%Y-%m-%d"), "model": name, "feature": feat, "views": multi, "dim": int(V.shape[1]), "n": int(V.shape[0]),
              "prep": {"size": SIZE, "margin": MARGIN, "mean": MEAN.tolist(), "std": STD.tolist(), "white_cut": 40},
              "heads": [[h[0], h[1]] for h, _ in got]}
    hj = json.dumps(header, separators=(",", ":")).encode()
    body = b"HDIX" + struct.pack("<I", len(hj)) + hj + V.astype("<f2").tobytes()
    (root / "heads-index.bin.gz").write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
    chosen = f"{name} {feat}" + (" +views" if multi else "")
    ev = {"built": header["built"], "heads": len(got), "tested": len(sample), "chosen": chosen, "model": name, "feature": feat,
          "exact": {k: results[chosen][k] for k in ("top1", "top5", "top10")}, "all": results}
    (root / "heads-eval.json").write_text(json.dumps(ev, indent=1))
    print("Chose", chosen)
    if name != "heads-model-full.onnx" and (root / "heads-model-full.onnx").exists():
        (root / "heads-model-full.onnx").unlink()          # not needed by the app


if __name__ == "__main__":
    main()
