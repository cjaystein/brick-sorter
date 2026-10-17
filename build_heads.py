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

MODEL_URL = "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx"
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


class Model:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        names = [o.name for o in self.s.get_outputs()]
        self.out = "pooler_output" if "pooler_output" in names else names[0]

    def __call__(self, x):
        y = self.s.run([self.out], {self.inp: x})[0]
        v = y[0] if y.ndim == 2 else y[0, 0]          # pooled, or the first (class) token
        return v / (np.linalg.norm(v) + 1e-9)


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
    args = ap.parse_args()
    root, cache = Path(args.root), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    model_path = root / args.model
    if not model_path.exists():
        print("Downloading the image model...")
        data = fetch(MODEL_URL)
        if not data:
            sys.exit("Couldn't download the image model")
        model_path.write_bytes(data)
    model = Model(model_path)

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

    got, vecs = [], []
    with cf.ThreadPoolExecutor(16) as ex:
        for i, (h, img) in enumerate(ex.map(get, heads)):
            if img is None:
                continue
            got.append((h, img)); vecs.append(model(prep(img)))
            if i % 500 == 0:
                print(f"  {i} of {len(heads)}")
    V = np.stack(vecs).astype(np.float32)
    header = {"built": time.strftime("%Y-%m-%d"), "model": args.model, "dim": int(V.shape[1]), "n": int(V.shape[0]),
              "prep": {"size": SIZE, "margin": MARGIN, "mean": MEAN.tolist(), "std": STD.tolist(), "white_cut": 40},
              "heads": [[h[0], h[1]] for h, _ in got]}
    hj = json.dumps(header, separators=(",", ":")).encode()
    body = b"HDIX" + struct.pack("<I", len(hj)) + hj + V.astype("<f2").tobytes()
    (root / "heads-index.bin.gz").write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
    print(f"Saved heads-index.bin.gz: {V.shape[0]} heads, {V.shape[1]} numbers each")

    # How well does it do on worse pictures? (the right head, and the right print in any colour)
    rng = random.Random(7)
    sample = rng.sample(range(len(got)), min(args.eval, len(got)))
    Vh = V.astype(np.float16).astype(np.float32)
    prints = [h[0].lower() for h, _ in got]
    ranks, pranks = [], []
    for i in sample:
        q = model(prep(worse(got[i][1], rng)))
        order = np.argsort(-(Vh @ q))
        ranks.append(int(np.nonzero(order == i)[0][0]) + 1)
        pranks.append(next(k + 1 for k, j in enumerate(order) if prints[j] == prints[i]))
    def share(rs, k): return round(sum(r <= k for r in rs) / len(rs), 3) if rs else 0
    ev = {"built": header["built"], "heads": len(got), "tested": len(sample),
          "exact": {"top1": share(ranks, 1), "top5": share(ranks, 5), "top10": share(ranks, 10)},
          "print_any_colour": {"top1": share(pranks, 1), "top5": share(pranks, 5), "top10": share(pranks, 10)}}
    (root / "heads-eval.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev))


if __name__ == "__main__":
    main()
