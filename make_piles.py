#!/usr/bin/env python3
"""
Make training photos of piles from photos of your parts spread out.

A detector only learns to pull a pile apart from examples of piles with a box
around every item. Drawing those by hand takes ages, so this makes them:

  1. extract   Cuts every item out of your spread-out photos (the same way the
               background method finds them), keeping its exact outline, and
               makes empty backgrounds by painting the items out.
  2. generate  Drops those cut-outs on top of each other on the backgrounds, in
               heaps, thousands of times. Because it places every item itself,
               it knows exactly where each one is, so every pile comes labelled.

    python make_piles.py extract spread/ --out piles
    (look at piles/cutouts_preview.jpg; delete any bad cut-outs in piles/cutouts/)
    python make_piles.py generate --out piles --count 4000

The result in piles/dataset is a YOLO dataset (data.yaml) for training, with
your real spread-out photos added as well. train_pile_detector.ipynb runs all
of this and the training.

Photos: parts spread out with a gap between each, on a plain background, shot
from above, about as far away as you'll shoot your piles. Flip parts over and
photograph them again to get both sides. Optional: photos of empty surfaces
(--backgrounds) so the detector learns more than one background.

Needs: pip install opencv-python numpy   (brick_sorter.py must be next to this file)
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brick_sorter import foreground_mask  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WORK_SIZE = 1600          # every photo is scaled so its longer side is this many pixels


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def read_photo(path):
    """Read a photo (turned the right way up) and scale it to WORK_SIZE."""
    img = cv2.imread(str(path))            # OpenCV applies the phone's rotation tag
    if img is None:
        return None
    s = WORK_SIZE / max(img.shape[:2])
    if s != 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return img


# --------------------------------------------------------------------------
# 1. extract: cut-outs and empty backgrounds
# --------------------------------------------------------------------------

def fill_holes(mask):
    """Fill enclosed holes (a printed face that matches the background, a stud hole)."""
    h, w = mask.shape
    flood = mask.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def cutouts_from_photo(img, sensitivity=1.0, min_frac=0.0003, max_frac=0.12):
    """Every separate item in a spread-out photo: [(bgra cut-out, (x0, y0, x1, y1))], plus the full mask."""
    h, w = img.shape[:2]
    # Joined mask (as the scanner uses) groups the bits of one item; the tight mask gives its outline.
    joined, _ = foreground_mask(img, sensitivity, analysis_size=max(h, w))
    tight, _ = foreground_mask(img, sensitivity, analysis_size=max(h, w), close_frac=0.0015)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_frac * h * w or bw * bh > max_frac * h * w:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                        # cut off by the edge of the photo
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255
        m = fill_holes(cv2.bitwise_and(comp, tight[y:y + bh, x:x + bw]))
        if cv2.countNonZero(m) < 0.4 * area:
            m = fill_holes(comp)
        # keep the largest piece only, then soften the edge by one pixel
        k, lab2, st2, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if k > 2:
            big = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
            m = (lab2 == big).astype(np.uint8) * 255
        alpha = cv2.GaussianBlur(m, (3, 3), 0)
        pad = 2
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
        a = np.zeros((y1 - y0, x1 - x0), np.uint8)
        a[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = alpha
        bgra = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
        bgra[..., 3] = a
        out.append((bgra, (int(x), int(y), int(x + bw), int(y + bh))))
    return out, joined


def empty_background(img, mask):
    """The photo with every item painted out."""
    grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    s = 0.25
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(grown, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_mask, 7, cv2.INPAINT_TELEA)
    filled = cv2.resize(cv2.GaussianBlur(filled, (5, 5), 0), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # put back a little of the surface's texture so it isn't perfectly smooth
    noise = np.random.default_rng(0).normal(0, 2.0, img.shape).astype(np.float32)
    filled = np.clip(filled.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(grown, (21, 21), 0).astype(np.float32)[..., None] / 255
    return (img * (1 - soft) + filled * soft).astype(np.uint8)


def contact_sheet(paths, out, cell=110, cols=14, limit=420):
    paths = paths[:limit]
    if not paths:
        return
    rows = math.ceil(len(paths) / cols)
    sheet = np.full((rows * (cell + 16), cols * cell, 3), 245, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        s = (cell - 8) / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        a = im[..., 3:4].astype(np.float32) / 255
        rgb = (im[..., :3] * a + 245 * (1 - a)).astype(np.uint8)
        r, c = divmod(i, cols)
        y, x = r * (cell + 16) + 4, c * cell + 4
        sheet[y:y + rgb.shape[0], x:x + rgb.shape[1]] = rgb
        cv2.putText(sheet, p.stem[-12:], (c * cell + 2, r * (cell + 16) + cell + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def extract(photos, out, backgrounds=None, sensitivity=1.0):
    """Cut every item out of the spread-out photos. Returns a summary dict."""
    out = Path(out)
    cut_dir, plate_dir, real_dir = out / "cutouts", out / "backgrounds", out / "real"
    for d in (cut_dir, plate_dir, real_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    files = list_images(photos)
    if not files:
        raise SystemExit(f"No photos found in {photos}. Put photos of your parts spread out there (JPG or PNG).")
    n_cut, real = 0, []
    for k, f in enumerate(files, 1):
        img = read_photo(f)
        if img is None:
            print(f"  can't read {f.name} (HEIC? export it as JPG)")
            continue
        items, mask = cutouts_from_photo(img, sensitivity)
        stem = f"{k:03d}_{f.stem[:30]}"
        for j, (bgra, _) in enumerate(items, 1):
            cv2.imwrite(str(cut_dir / f"{stem}_{j:03d}.png"), bgra)
        n_cut += len(items)
        cv2.imwrite(str(plate_dir / f"{stem}.jpg"), empty_background(img, mask), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(real_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        real.append({"image": f"{stem}.jpg", "boxes": [b for _, b in items], "size": [img.shape[1], img.shape[0]]})
        print(f"  {f.name}: {len(items)} items")
    for f in list_images(backgrounds) if backgrounds else []:
        img = read_photo(f)
        if img is not None:
            cv2.imwrite(str(plate_dir / f"empty_{f.stem[:30]}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "real_boxes.json").write_text(json.dumps(real))
    contact_sheet(sorted(cut_dir.glob("*.png")), out / "cutouts_preview.jpg")
    n_plates = len(list(plate_dir.glob("*.jpg")))
    print(f"\n{n_cut} cut-outs from {len(real)} photos, {n_plates} backgrounds.")
    print(f"Check {out / 'cutouts_preview.jpg'}. Delete any cut-out in {cut_dir} that is two parts stuck together, "
          "a shadow or a speck, then run generate.")
    if n_cut < 150:
        print("That's not many: aim for 300 or more different items for a detector that copes with your whole collection.")
    return {"cutouts": n_cut, "photos": len(real), "backgrounds": n_plates}


# --------------------------------------------------------------------------
# 2. generate: labelled piles
# --------------------------------------------------------------------------

def _rotate_bgra(im, angle, scale):
    h, w = im.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(im, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _jitter_plate(plate, rng):
    if rng.random() < 0.5:
        plate = cv2.flip(plate, 1)
    if rng.random() < 0.5:
        plate = cv2.flip(plate, 0)
    if rng.random() < 0.3:
        plate = cv2.rotate(plate, cv2.ROTATE_180)
    return plate


def make_pile(plate, cutouts, rng, visible_min=0.3, min_px=150):
    """One pile on a background: (image, [(x0, y0, x1, y1)])."""
    img = _jitter_plate(plate, rng).astype(np.float32)
    H, W = img.shape[:2]
    ids = np.zeros((H, W), np.int32)
    zoom = rng.uniform(0.75, 1.3)                    # camera a little closer or further than your spread photos
    density = rng.uniform(0.15, 0.7)                 # share of the photo covered by items
    clusters = [(rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H) for _ in range(rng.integers(1, 4))]
    light = rng.uniform(0, 2 * math.pi)
    shadow_dx, shadow_dy = math.cos(light), math.sin(light)
    shadow_strength = rng.uniform(0.15, 0.45)

    placed, area, target = [], 0.0, density * W * H
    spread = math.sqrt(target / len(clusters)) * rng.uniform(0.35, 0.6)
    tries = 0
    while area < target and len(placed) < 90 and tries < 400:
        tries += 1
        cut = cutouts[rng.integers(len(cutouts))]
        item = _rotate_bgra(cut, rng.uniform(0, 360), zoom * rng.uniform(0.92, 1.08))
        ih, iw = item.shape[:2]
        if ih >= H * 0.9 or iw >= W * 0.9:
            continue
        cx, cy = clusters[rng.integers(len(clusters))]
        x = int(np.clip(rng.normal(cx, spread) - iw / 2, -iw * 0.3, W - iw * 0.7))
        y = int(np.clip(rng.normal(cy, spread) - ih / 2, -ih * 0.3, H - ih * 0.7))
        # visible part of the item inside the photo
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + iw), min(H, y + ih)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        sub = item[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
        a = sub[..., 3:4] / 255
        rgb = sub[..., :3] * rng.uniform(0.85, 1.12) + rng.uniform(-8, 8)
        # soft shadow first, offset away from the light
        r = max(3, int(0.06 * max(ih, iw)))
        sh = cv2.GaussianBlur(item[..., 3], (2 * r + 1, 2 * r + 1), 0).astype(np.float32) / 255
        off = int(0.04 * max(ih, iw)) + 1
        M = np.float32([[1, 0, shadow_dx * off], [0, 1, shadow_dy * off]])
        sh = cv2.warpAffine(sh, M, (iw, ih))[y0 - y:y1 - y, x0 - x:x1 - x]
        img[y0:y1, x0:x1] *= (1 - shadow_strength * sh)[..., None]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + np.clip(rgb, 0, 255) * a
        full = int((item[..., 3] > 127).sum())
        k = len(placed) + 1
        ids[y0:y1, x0:x1][a[..., 0] > 0.5] = k
        placed.append(full)
        area += full

    boxes = []
    for k, full in enumerate(placed, 1):
        ys, xs = np.nonzero(ids == k)
        if len(xs) < min_px or len(xs) < visible_min * full:
            continue                                     # mostly hidden under other items: not labelled
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    # whole-photo lighting and camera effects
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    grad = 1 + rng.uniform(0, 0.18) * ((gx / W - 0.5) * math.cos(ang) + (gy / H - 0.5) * math.sin(ang))
    img *= grad[..., None]
    img *= np.array([rng.uniform(0.94, 1.06), 1.0, rng.uniform(0.94, 1.06)], np.float32)   # colour of the light
    img += rng.normal(0, rng.uniform(0, 4), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, boxes


def _yolo_lines(boxes, w, h):
    return "".join(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"
                   for x0, y0, x1, y1 in boxes)


_CUTS, _PLATES = None, None


def _init(cut_files, plate_files):
    global _CUTS, _PLATES
    _CUTS = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in cut_files]
    _CUTS = [c for c in _CUTS if c is not None and c.ndim == 3 and c.shape[2] == 4]
    _PLATES = [cv2.imread(str(p)) for p in plate_files]
    _PLATES = [p for p in _PLATES if p is not None]


def _one(job):
    i, seed, split, out = job
    rng = np.random.default_rng(seed)
    plate = _PLATES[rng.integers(len(_PLATES))]
    img, boxes = make_pile(plate, _CUTS, rng)
    name = f"pile_{i:05d}"
    cv2.imwrite(str(Path(out) / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(72, 95))])
    (Path(out) / "labels" / split / f"{name}.txt").write_text(_yolo_lines(boxes, img.shape[1], img.shape[0]))
    return len(boxes)


def draw_boxes(img, boxes):
    out = img.copy()
    t = max(2, img.shape[1] // 500)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 190, 255), t)
    return out


def generate(out, count=4000, val_fraction=0.1, seed=1, workers=None, include_real=True, extra_real=None):
    """Make `count` labelled piles in out/dataset (YOLO format). Returns the data.yaml path.
    extra_real: folders of real labelled photos to add (images/*.jpg with labels/*.txt in YOLO format),
    e.g. the "detection" folder of a training zip downloaded from the Brick Sorter web app."""
    out = Path(out)
    cut_files = sorted((out / "cutouts").glob("*.png"))
    plate_files = sorted((out / "backgrounds").glob("*.jpg"))
    if len(cut_files) < 10 or not plate_files:
        raise SystemExit(f"Run extract first: found {len(cut_files)} cut-outs and {len(plate_files)} backgrounds in {out}.")
    ds = out / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)

    n_val = max(1, int(count * val_fraction))
    jobs = [(i, seed * 1_000_003 + i, "val" if i < n_val else "train", str(ds)) for i in range(count)]
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    print(f"Making {count} piles from {len(cut_files)} cut-outs on {len(plate_files)} backgrounds ({workers} workers)...")
    total = 0
    done = 0
    if workers > 1:
        try:
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(cut_files, plate_files)) as pool:
                for n in pool.map(_one, jobs, chunksize=16):
                    total += n
                    done += 1
                    if done % 250 == 0 or done == count:
                        print(f"  {done} of {count}")
        except Exception as e:                     # e.g. Windows notebooks that can't start worker processes
            print(f"  Several processes didn't work here ({type(e).__name__}); carrying on with one, which is slower.")
    if done < count:
        _init(cut_files, plate_files)
        for job in jobs[done:]:
            total += _one(job)
            done += 1
            if done % 250 == 0 or done == count:
                print(f"  {done} of {count}")

    n_real = 0
    real_file = out / "real_boxes.json"
    if include_real and real_file.exists():
        for r in json.loads(real_file.read_text()):          # your real photos, as easy examples
            src = out / "real" / r["image"]
            if src.exists() and r["boxes"]:
                shutil.copy(src, ds / "images" / "train" / f"real_{src.name}")
                (ds / "labels" / "train" / f"real_{src.stem}.txt").write_text(_yolo_lines(r["boxes"], *r["size"]))
                n_real += 1

    n_app = 0
    for k, folder in enumerate(extra_real or []):            # photos you confirmed in the web app
        folder = Path(folder)
        for img in list_images(folder / "images"):
            lab = folder / "labels" / f"{img.stem}.txt"
            if lab.exists() and lab.read_text().strip():
                shutil.copy(img, ds / "images" / "train" / f"app{k}_{img.name}")
                shutil.copy(lab, ds / "labels" / "train" / f"app{k}_{img.stem}.txt")
                n_app += 1
    (ds / "data.yaml").write_text(f"path: {ds.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n")
    # preview of the first few piles with their boxes
    tiles = []
    for p in sorted((ds / "images" / "train").glob("pile_*.jpg"))[:6]:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        boxes = []
        for line in (ds / "labels" / "train" / f"{p.stem}.txt").read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
        tiles.append(cv2.resize(draw_boxes(im, boxes), (640, int(640 * h / w))))
    if tiles:
        th = min(t.shape[0] for t in tiles)
        tiles = [t[:th] for t in tiles]
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) // 3 * 3, 3)] or [tiles[0]]
        if rows:
            cv2.imwrite(str(out / "piles_preview.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\nMade {count} piles with {total} labelled items (about {total // max(1, count)} per pile), "
          f"plus {n_real} of your real photos{f' and {n_app} labelled photos from the web app' if n_app else ''}. Dataset: {ds / 'data.yaml'}")
    print(f"Preview: {out / 'piles_preview.jpg'}")
    return ds / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="cut items out of photos of parts spread out")
    e.add_argument("photos", help="folder of photos of parts spread out")
    e.add_argument("--out", default="piles", help="working folder (default: piles)")
    e.add_argument("--backgrounds", help="optional folder of photos of empty surfaces")
    e.add_argument("--sensitivity", type=float, default=1.0, help="as in brick_sorter.py scan")
    g = sub.add_parser("generate", help="make labelled pile photos from the cut-outs")
    g.add_argument("--out", default="piles")
    g.add_argument("--count", type=int, default=4000, help="how many piles to make (default 4000)")
    g.add_argument("--workers", type=int, default=None, help="processes to use (default: CPU cores - 1, at most 8)")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--extra", nargs="*", default=[], help="folders of labelled photos to add, e.g. the detection folder from the web app's training zip")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.photos, args.out, args.backgrounds, args.sensitivity)
    else:
        generate(args.out, args.count, seed=args.seed, workers=args.workers, extra_real=args.extra)


if __name__ == "__main__":
    main()
