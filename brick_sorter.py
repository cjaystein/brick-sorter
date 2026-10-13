#!/usr/bin/env python3
"""
Brick Sorter - identify LEGO parts and minifigures from bulk photos and check
them against sets.

  scan     Find every separate item in one or more photos, crop each one,
           identify it with Brickognize and add it to your inventory CSV.
  name-fig Work out which minifigure some loose parts (head, torso, legs) belong to.
  value    Price your inventory with BrickLink's price guide.
  compare  Fetch sets' parts lists and minifigures from Rebrickable and report
           what you have and what's missing, as an Excel workbook.

Photo tips: spread parts out on a plain, contrasting background (white paper
for dark parts, dark cloth for white or light parts), keep them apart so they
don't touch, and shoot from directly above in even light.

Setup:
  pip install opencv-python-headless numpy requests openpyxl

Examples:
  python brick_sorter.py scan photos/ --inventory inventory.csv
  python brick_sorter.py scan tray1.jpg tray2.jpg --debug
  python brick_sorter.py scan pile.jpg --model lego-detector.pt   # trained detector
  python brick_sorter.py scan shelf.jpg --mode figs --model yolo11n.pt   # minifigures, even on shelves
  python brick_sorter.py scan photos/ --sets 10696 6177 --key YOUR_REBRICKABLE_KEY
  python brick_sorter.py scan figs/ --mode figs --sets 10696 6177 --key YOUR_REBRICKABLE_KEY
  python brick_sorter.py compare 10696 --inventory inventory.csv --key YOUR_REBRICKABLE_KEY
  python brick_sorter.py compare 10696 6177 31120 --key YOUR_REBRICKABLE_KEY
  python brick_sorter.py suggest --key YOUR_REBRICKABLE_KEY      # which sets is this from?
  python brick_sorter.py name-fig loose_parts.jpg               # which minifigure are these parts from?
  python brick_sorter.py value --currency AUD                   # BrickLink value of your inventory

After a scan, scan_review.html opens with each crop next to the matched
part's picture and BrickLink / Rebrickable links, so you can check the guesses.
BrickLink prices (scan --prices, value, name-fig --prices) need your BrickLink
API keys in bricklink_keys.json next to this script; see the README.

--sets on scan: tell it which sets you think are in the pile. Their parts
lists steer the guesses: a close second guess that is in your sets beats a top
guess that isn't, and colours are limited to the colours that part comes in
within your sets. Parts outside your sets are still accepted, but flagged.

Minifigures: photograph them assembled, lying face up and spread out, and use
--mode figs (or --mode mixed for photos with parts and figures together).
compare checks each set's minifigures as whole figures. Brickognize and
Rebrickable number minifigures differently, so a scanned figure is linked to a
set's figure by --sets (name match, saved in the rb_fig column), by the same
ID, or by a similar name at compare time (marked "check"). You can fix a link
by typing the Rebrickable figure number (e.g. fig-001234) into rb_fig.

The Rebrickable key can also be set in the REBRICKABLE_API_KEY environment
variable. Get one free at https://rebrickable.com/api/ (account > Settings > API).

inventory.csv has one row per detected piece. Open it in a spreadsheet to fix
wrong guesses (edit part_id / color_name) or delete rows, then run compare.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

BRICKOGNIZE_BASE = os.environ.get("BRICK_SORTER_BG_URL", "https://api.brickognize.com")
REBRICKABLE_URL = os.environ.get("BRICK_SORTER_RB_URL", "https://rebrickable.com/api/v3/lego")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}

INVENTORY_FIELDS = [
    "type", "part_id", "part_name", "color_name", "color_id", "quantity",
    "match_score", "color_score", "needs_review", "alternatives",
    "bricklink_id", "rb_part", "rb_fig", "note", "source_photo", "crop_file", "listing_id",
    "img_url", "price_each", "currency",
]


# --------------------------------------------------------------------------
# Detection: split one photo of many parts into one crop per part
# --------------------------------------------------------------------------

def foreground_mask(image_bgr, sensitivity=1.0, analysis_size=1000, close_frac=0.006):
    """Mask (uint8, 255 = not background) at analysis size, and the scale used.

    See detect_parts. close_frac sets how far apart bits of one part may be and
    still be joined (as a fraction of the photo's longer side).
    """
    h, w = image_bgr.shape[:2]
    scale = analysis_size / max(h, w)
    small = cv2.resize(image_bgr, (round(w * scale), round(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr.copy()
    scale = min(scale, 1.0)
    sh, sw = small.shape[:2]

    small = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV 8-bit Lab: L is 0-255, a/b are offset by 128. Rescale to real Lab units.
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0

    # Background model: the photo is split into a coarse grid and each cell's
    # median colour is taken as the local background. That follows uneven
    # lighting across the sheet. A second pass ignores the parts found in the
    # first, so cells crowded with parts don't pull the background towards them.
    def background(fg=None, cells=12):
        gy, gx = cells, max(1, round(cells * sw / sh))
        grid = np.zeros((gy, gx, 3), np.float32)
        ok = np.zeros((gy, gx), bool)
        for j in range(gy):
            for i in range(gx):
                y0, y1 = j * sh // gy, (j + 1) * sh // gy
                x0, x1 = i * sw // gx, (i + 1) * sw // gx
                px = lab[y0:y1, x0:x1].reshape(-1, 3)
                if fg is not None:
                    px = px[~fg[y0:y1, x0:x1].ravel()]
                if len(px) > 0.2 * (y1 - y0) * (x1 - x0):
                    grid[j, i] = np.median(px, axis=0)
                    ok[j, i] = True
        fill = np.median(grid[ok], axis=0) if ok.any() else np.median(lab.reshape(-1, 3), axis=0)
        grid[~ok] = fill
        for c in range(3):
            grid[..., c] = cv2.medianBlur(grid[..., c].copy(), 3)
        return cv2.resize(grid, (sw, sh), interpolation=cv2.INTER_LINEAR)

    def difference(bg):
        # Lighting changes L far more than colour, so colour differences weigh more.
        d_l = np.abs(lab[..., 0] - bg[..., 0])
        d_ab = np.sqrt(((lab[..., 1:] - bg[..., 1:]) ** 2).sum(axis=2))
        return np.maximum(d_ab * 1.4, d_l * 0.8)

    def threshold(diff, fg=None):
        # Noise level of the background itself (median + 6 x robust spread),
        # so grey and dark parts count even when bright parts are in the photo.
        vals = diff[~fg] if fg is not None else diff.ravel()
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        return max(6.0, med + 6.0 * mad) / max(sensitivity, 0.1)

    diff = difference(background())
    first = diff > threshold(diff)
    first = cv2.dilate(first.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    diff = difference(background(first))
    thresh = threshold(diff, first)
    mask = (diff > thresh).astype(np.uint8) * 255

    r = max(2, int(close_frac * max(sh, sw)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask, scale


def detect_parts(image_bgr, sensitivity=1.0, min_size=0.0003, analysis_size=1000):
    """Return a list of (x0, y0, x1, y1) boxes in full-resolution pixels.

    The local background colour is estimated across the photo. Every pixel
    that differs enough from it (in Lab colour space) is foreground. Nearby
    foreground is joined, small specks are dropped, and each remaining blob
    becomes one part.

    sensitivity: higher finds fainter parts (and more shadows); 0.5-2.
    min_size:    smallest part kept, as a fraction of the photo's area.
    """
    h, w = image_bgr.shape[:2]
    mask, scale = foreground_mask(image_bgr, sensitivity, analysis_size)
    sh, sw = mask.shape[:2]

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = min_size * sh * sw
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if bw * bh > 0.6 * sh * sw:          # almost the whole photo: background, not a part
            continue
        pad = int(0.12 * max(bw, bh)) + 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(sw, x + bw + pad), min(sh, y + bh + pad)
        boxes.append((int(x0 / scale), int(y0 / scale),
                      min(w, int(x1 / scale)), min(h, int(y1 / scale))))
    boxes.sort(key=lambda b: (b[1] // max(1, int(0.08 * h)), b[0]))  # reading order
    return boxes


_MODEL_CACHE = {}


def _tile_rects(w, h, n, overlap=0.25):
    if n <= 1:
        return [(0, 0, w, h)]
    tw, th = w / (n - (n - 1) * overlap), h / (n - (n - 1) * overlap)
    sx, sy = tw * (1 - overlap), th * (1 - overlap)
    return [(round(i * sx), round(j * sy), round(min(w, i * sx + tw)), round(min(h, j * sy + th)))
            for j in range(n) for i in range(n)]


def detect_parts_model(image_bgr, model_path, conf=None, imgsz=960, tiles=None):
    """Find items with a YOLO model: your brick detector (train_lego_detector.ipynb),
    or the figure finder (a general model such as yolo11n.pt or figure-finder.onnx,
    which finds minifigures as "person" and searches 3 x 3 zoomed-in tiles too).
    Returns (x0, y0, x1, y1) boxes in full-resolution pixels.
    """
    try:
        import torch
        from torchvision.ops import nms
        from ultralytics import YOLO
    except ImportError:
        sys.exit("--model needs the ultralytics package: pip install ultralytics")
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path, task="detect")
    model = _MODEL_CACHE[model_path]
    people = [k for k, v in model.names.items() if v == "person"]
    figures = bool(people) and "brick" not in model.names.values()
    classes = people if figures else None
    if conf is None:
        conf = 0.1 if figures else 0.3
    if tiles is None:
        tiles = 3 if figures else 1
    iou = 0.5 if figures else 0.7

    h, w = image_bgr.shape[:2]
    regions = [(0, 0, w, h)] + (_tile_rects(w, h, tiles) if tiles > 1 else [])
    boxes, scores = [], []
    for x0, y0, x1, y1 in regions:
        r = model.predict(image_bgr[y0:y1, x0:x1], conf=conf, imgsz=imgsz, iou=iou, agnostic_nms=True,
                          classes=classes, max_det=1000, verbose=False)[0]
        whole = (x0, y0, x1, y1) == regions[0]
        for (a, b, c, d), sc in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()):
            if not whole and ((a < 2 and x0 > 0) or (b < 2 and y0 > 0) or
                              (c > (x1 - x0) - 2 and x1 < w) or (d > (y1 - y0) - 2 and y1 < h)):
                continue            # cut off by a tile edge; found whole in a neighbouring tile
            if tiles > 1 and (c - a) * (d - b) > 0.08 * w * h:
                continue            # far bigger than one figure
            boxes.append([a + x0, b + y0, c + x0, d + y0])
            scores.append(sc)
    if not boxes:
        return []
    kept = [boxes[i] for i in nms(torch.tensor(boxes), torch.tensor(scores), iou).tolist()]
    if tiles > 1:               # drop boxes mostly inside a bigger one (a head found apart from its figure)
        area = lambda q: max(0, q[2] - q[0]) * max(0, q[3] - q[1])
        inter = lambda q, o: max(0, min(q[2], o[2]) - max(q[0], o[0])) * max(0, min(q[3], o[3]) - max(q[1], o[1]))
        kept = [q for q in kept if not any(o is not q and area(o) > area(q) and inter(q, o) > 0.8 * area(q) for o in kept)]
    out = []
    for x0, y0, x1, y1 in kept:
        pad = 0.08 * max(x1 - x0, y1 - y0) + 4
        out.append((max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(w, int(x1 + pad)), min(h, int(y1 + pad))))
    out.sort(key=lambda b: (b[1] // max(1, int(0.08 * h)), b[0]))
    return out


def save_debug(image_bgr, boxes, path):
    out = image_bgr.copy()
    t = max(2, int(0.003 * max(out.shape[:2])))
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), t)
        cv2.putText(out, str(i), (x0 + t, y0 + 12 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t, (0, 200, 255), t)
    cv2.imwrite(str(path), out)


# --------------------------------------------------------------------------
# Brickognize
# --------------------------------------------------------------------------

ENDPOINTS = {"parts": "/predict/parts/", "figs": "/predict/figs/", "mixed": "/predict/"}
WHAT = {"parts": "parts", "figs": "minifigures", "mixed": "items"}


def identify(crop_path, session, mode="parts", retries=3):
    params = {"top_k_items": 5}
    if mode != "figs":
        params.update(predict_color="true", top_k_colors=3)
    url = BRICKOGNIZE_BASE + ENDPOINTS[mode]
    for attempt in range(retries):
        try:
            with open(crop_path, "rb") as f:
                r = session.post(url, params=params,
                                 files={"query_image": (crop_path.name, f, "image/jpeg")},
                                 timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(2 * (attempt + 1))
    return {"error": "Brickognize kept returning errors"}


def bricklink_id(item):
    for site in item.get("external_sites") or []:
        if site.get("name", "").lower() == "bricklink":
            m = re.search(r"[?&][PMS]=([^&]+)", site.get("url", ""))
            if m:
                return requests.utils.unquote(m.group(1))
    return item.get("id", "")


def item_type(item, mode="parts"):
    return item.get("type") or ("fig" if mode == "figs" else "part")


def row_type(row):
    return (row.get("type") or "part").strip().lower() or "part"


STOP_WORDS = {"minifig", "minifigure", "figure", "with", "and", "the", "a", "an", "of", "in", "on", "lego"}


def name_tokens(name):
    words = re.findall(r"[a-z0-9]+", (name or "").lower().replace("grey", "gray"))
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}


def name_similarity(a, b):
    """0..1 overlap of the meaningful words in two minifigure names."""
    A, B = name_tokens(a), name_tokens(b)
    if not A or not B:
        return 0.0
    return 2 * len(A & B) / (len(A) + len(B))


FIG_LINK_AT = 0.5      # name overlap needed to link a scanned figure to a set's figure
FIG_SURE_AT = 0.75     # below this the link is flagged for checking


def to_row(result, photo, crop, review_below, catalogue=None, mode="parts"):
    base = {"quantity": 1, "source_photo": photo, "crop_file": crop,
            "listing_id": result.get("listing_id", "")}
    if "error" in result:
        return dict(base, part_name=f"ERROR: {result['error']}", needs_review="yes")
    items = result.get("items") or []
    colors = result.get("colors") or []
    if not items:
        return dict(base, part_name="No match", needs_review="yes")

    if catalogue:
        pick = catalogue.choose(items, colors, mode)
    else:
        it = items[0]
        kind = item_type(it, mode)
        col = (colors[0] if colors else {}) if kind == "part" else {}
        pick = {"item": it, "kind": kind, "entry": None, "fig": None, "fig_sim": 0,
                "color_name": col.get("name", ""), "color_id": col.get("id", ""), "notes": []}
    it, kind = pick["item"], pick["kind"]
    score = float(it.get("score", 0))
    notes = list(pick["notes"])

    if kind == "part":
        if catalogue:
            # A match to one of your sets is strong evidence, so only weak or
            # out-of-set guesses, or a colour your sets don't have it in, need a look.
            review = pick["entry"] is None or score < 0.4 or not pick["color_name"] or pick.get("colour_mismatch")
        else:
            review = score < review_below or not pick["color_name"]
    elif kind == "fig":
        if catalogue:
            review = pick["fig"] is None or pick["fig_sim"] < FIG_SURE_AT or score < 0.4
        else:
            review = score < review_below
    else:
        review = True
        notes.append(f"looks like a whole {kind}, not a part or minifigure")

    color_score = next((float(c.get("score", 0)) for c in colors
                        if norm_color(c.get("name")) == norm_color(pick["color_name"])), None)
    return dict(
        base,
        type=kind,
        part_id=it.get("id", ""),
        part_name=it.get("name", ""),
        color_name=pick["color_name"],
        color_id=pick["color_id"],
        match_score=f"{score:.2f}",
        color_score=f"{color_score:.2f}" if color_score is not None and pick["color_name"] else "",
        needs_review="yes" if review else "",
        alternatives="; ".join([f"{i.get('id')} {i.get('name')} ({float(i.get('score', 0)):.2f})"
                                for i in items if i is not it][:3]),
        bricklink_id=bricklink_id(it),
        img_url=it.get("img_url", ""),
        rb_part=pick["entry"]["part_num"] if pick["entry"] else "",
        rb_fig=pick["fig"]["fig_num"] if pick["fig"] else "",
        note="; ".join(notes),
    )


def read_inventory(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_inventory(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in INVENTORY_FIELDS})


def cmd_scan(args):
    catalogue = None
    if args.sets:
        key = args.key or os.environ.get("REBRICKABLE_API_KEY")
        if not key:
            sys.exit("--sets needs a Rebrickable API key: --key or REBRICKABLE_API_KEY.")
        catalogue = SetCatalogue.fetch(args.sets, key)
        for info, lines, figs in catalogue.sets:
            n_figs = sum(f["quantity"] for f in figs)
            print(f"Expecting {info['set_num']} {info.get('name', '')} "
                  f"({sum(l['quantity'] for l in lines)} pieces"
                  f"{f', {n_figs} minifigures' if n_figs else ''})")

    photos = []
    for p in args.photos:
        p = Path(p)
        if p.is_dir():
            photos += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.exists():
            photos.append(p)
        else:
            print(f"Not found: {p}", file=sys.stderr)
    if not photos:
        sys.exit("No photos to scan.")

    crops_dir = Path(args.crops)
    crops_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for photo in photos:
        img = cv2.imread(str(photo))
        if img is None:
            print(f"Can't read {photo} (HEIC photos need converting to JPEG first)", file=sys.stderr)
            continue
        if args.model:
            boxes = detect_parts_model(img, args.model, conf=args.conf, imgsz=args.imgsz, tiles=args.tiles)
        else:
            boxes = detect_parts(img, sensitivity=args.sensitivity, min_size=args.min_size)
        print(f"{photo.name}: found {len(boxes)} {WHAT[args.mode]}")
        if args.debug:
            save_debug(img, boxes, crops_dir / f"{photo.stem}_detected.jpg")
        for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
            crop = img[y0:y1, x0:x1]
            s = 640 / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            out = crops_dir / f"{photo.stem}_{i:03d}.jpg"
            cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jobs.append((photo.name, out))

    if args.detect_only:
        print(f"Saved {len(jobs)} crops to {crops_dir}/ (no identification run).")
        return

    print(f"Identifying {len(jobs)} {WHAT[args.mode]} with Brickognize...")
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda j: identify(j[1], session, args.mode), jobs))

    rows = read_inventory(args.inventory)
    new = [to_row(res, photo, str(crop), args.review_below, catalogue, args.mode)
           for (photo, crop), res in zip(jobs, results)]
    key = args.key or os.environ.get("REBRICKABLE_API_KEY")
    cache_path = Path(args.inventory).resolve().with_name(".brick_sorter_cache.json")
    cache = load_cache(cache_path)
    rb_colors = rb_colours(key, cache)
    save_cache(cache_path, cache)
    if args.prices:
        add_prices([r for r in new if r.get("part_id")], args, rb_colors)
    write_inventory(args.inventory, rows + new)
    print_scan_summary(new, args.inventory, catalogue)
    page = Path(args.inventory).resolve().with_name("scan_review.html")
    write_review_html(page, new, "Check the matches",
                      f"{len(new)} items from {len({p for p, _ in jobs})} photo(s). Each of your crops is next to the matched "
                      "part's picture; open BrickLink or Rebrickable to compare. Fix any wrong rows in "
                      f"{Path(args.inventory).name}.", rb_colors)
    print(f"Saved {page.name}: your crops next to the matches, with BrickLink and Rebrickable links.")
    open_page(page, args)


def print_scan_summary(new, inventory_path, catalogue):
    flagged = sum(1 for r in new if r.get("needs_review"))
    for r in new:
        flag = "  <- check" if r.get("needs_review") else ""
        what = r.get("color_name", "") if r.get("type", "part") == "part" else f"[{r.get('type')}]"
        print(f"  {Path(r['crop_file']).name:24} {r.get('part_id', ''):10} "
              f"{what:20} {r.get('part_name', '')[:40]}{flag}")
        if r.get("note"):
            print(f"  {'':24} {r['note']}")
    n_figs = sum(1 for r in new if r.get("type") == "fig")
    n_parts = sum(1 for r in new if r.get("type") == "part")
    unknown = sum(1 for r in new if not r.get("type"))
    msg = (f"\nAdded {n_parts} pieces"
           f"{f' and {n_figs} minifigures' if n_figs else ''} to {inventory_path} ({flagged} need checking"
           f"{f', including {unknown} not recognised' if unknown else ''}).")
    if catalogue:
        in_sets = sum(1 for r in new if r.get("rb_part") or r.get("rb_fig"))
        changed = sum(1 for r in new if (r.get("rb_part") or r.get("rb_fig")) and r.get("note"))
        msg += f" {in_sets} matched your expected sets; {changed} guesses were corrected or linked using them."
    print(msg)


# --------------------------------------------------------------------------
# Checking matches: links, reference pictures, BrickLink prices
# --------------------------------------------------------------------------

try:
    import bricklink        # optional, next to this script: live BrickLink prices and figure lookups
except ImportError:
    bricklink = None

# BrickLink colour numbers for common colours, used when Rebrickable's colour list isn't available.
BL_COLOURS = {
    "white": 1, "tan": 2, "yellow": 3, "orange": 4, "red": 5, "green": 6, "blue": 7, "brown": 8, "lightgray": 9,
    "darkgray": 10, "black": 11, "transclear": 12, "transblack": 13, "transdarkblue": 14, "translightblue": 15,
    "transneongreen": 16, "transred": 17, "transneonorange": 18, "transyellow": 19, "transgreen": 20, "chromegold": 21,
    "nougat": 28, "lime": 34, "brightgreen": 36, "darkturquoise": 39, "mediumblue": 42, "darkpink": 47, "sandgreen": 48,
    "sandblue": 55, "darkred": 59, "darkblue": 63, "darkorange": 68, "darktan": 69, "magenta": 71, "darkgreen": 80,
    "darkbluishgray": 85, "lightbluishgray": 86, "reddishbrown": 88, "darkpurple": 89, "lightnougat": 90,
    "flatsilver": 95, "brightlightyellow": 103, "brightpink": 104, "brightlightblue": 105, "brightlightorange": 110,
    "pearlgold": 115, "darkbrown": 120, "mediumnougat": 150, "darkazure": 153, "lavender": 154, "olivegreen": 155,
    "mediumazure": 156, "mediumlavender": 157, "yellowishgreen": 158, "coral": 220,
}
BL_LETTER = {"part": "P", "fig": "M", "set": "S"}


def bl_colour(name, rb_colors=None):
    """BrickLink colour number for a colour name (Rebrickable or BrickLink spelling)."""
    if not name:
        return None
    n = norm_color(name)
    for c in rb_colors or []:
        bl = (c.get("external_ids") or {}).get("BrickLink") or {}
        names = [c.get("name", "")] + [x for grp in bl.get("ext_descrs") or [] for x in grp]
        if bl.get("ext_ids") and any(norm_color(x) == n for x in names):
            return bl["ext_ids"][0]
    return BL_COLOURS.get(n)


def bl_url(kind, no, blc=None, name=""):
    t = BL_LETTER.get(kind, "P")
    if t == "M" and str(no).lower().startswith("fig-"):          # a Rebrickable-only figure number
        return f"https://www.bricklink.com/v2/search.page?q={requests.utils.quote(name or '')}#T=M"
    url = f"https://www.bricklink.com/v2/catalog/catalogitem.page?{t}={requests.utils.quote(str(no))}"
    return url + (f"#T=C&C={blc}" if t == "P" and blc is not None else "")


def rb_url(kind, num, name=""):
    q = requests.utils.quote
    if kind == "fig":
        return (f"https://rebrickable.com/minifigs/{q(num)}/" if str(num).lower().startswith("fig-")
                else f"https://rebrickable.com/search/?show_printed=on&search_type=minifig&q={q(name or num)}")
    if kind == "set":
        return f"https://rebrickable.com/sets/{q(num)}/"
    return f"https://rebrickable.com/search/?show_printed=on&search_type=part&q={q(num)}"


def bl_ready(explain=False):
    """True when bricklink.py and your BrickLink keys are set up."""
    why = ""
    if bricklink is None:
        why = "bricklink.py isn't next to brick_sorter.py."
    else:
        try:
            if not bricklink.configured():
                why = "BrickLink keys aren't set up: add bricklink_keys.json next to brick_sorter.py (see the README)."
        except bricklink.BrickLinkError as e:
            why = str(e)
    if why and explain:
        print(why, file=sys.stderr)
    return not why


def rb_colours(key, cache=None):
    """Rebrickable's colour list (with BrickLink numbers), remembered in the cache."""
    if not key:
        return None
    if cache is not None and cache.get("rb_colors"):
        return cache["rb_colors"]
    try:
        colors = rb_get_all(f"{REBRICKABLE_URL}/colors/", key)
    except SystemExit:
        return None
    if cache is not None:
        cache["rb_colors"] = colors
    return colors


def price_row(row, condition="U", guide="sold", currency=None, rb_colors=None):
    """BrickLink price guide for an inventory row: (price dict with 'each', reason if none)."""
    kind = row_type(row)
    no = (row.get("bricklink_id") or row.get("part_id") or "").strip()
    if kind not in ("part", "fig", "set") or not no or no.lower().startswith("fig-"):
        return None, "not a BrickLink number"
    blc = bl_colour(row.get("color_name"), rb_colors) if kind == "part" else None
    if kind == "part" and blc is None:
        return None, "colour unknown"
    try:
        p = bricklink.price(kind, no, blc, condition, guide, currency)
    except bricklink.BrickLinkError as e:
        return None, str(e)
    each = p["qty_avg"] if p.get("qty_avg") else p.get("avg")
    if not each:
        return None, "no sales in the last 6 months" if guide == "sold" else "none for sale"
    return dict(p, each=each), ""


def money(v, cur=""):
    if v is None or v == "":
        return ""
    v = float(v)
    return f"{v:,.3f} {cur}".strip() if v < 1 else f"{v:,.2f} {cur}".strip()


def _thumb_uri(path, size=200):
    """A small JPEG of a crop as a data: URI, so the page works when moved or opened on a phone."""
    import base64
    img = cv2.imread(str(path))
    if img is None:
        return None
    s = size / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def _rel(target, page):
    try:
        return Path(os.path.relpath(Path(target).resolve(), Path(page).resolve().parent)).as_posix()
    except ValueError:                      # another drive on Windows
        return Path(target).resolve().as_uri()


PAGE_CSS = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#18202b; --muted:#5d6a7b; --line:#dfe4ea; --accent:#1d5fd1; --warn:#9a6200; --warnbg:#fdf1d8; --good:#23855a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f141b; --card:#171e28; --ink:#e7ecf2; --muted:#9aa6b6; --line:#29333f; --accent:#6ea3f2; --warn:#e6b450; --warnbg:#3a2c12; --good:#5cc98a; } }
:root[data-theme="dark"] { --bg:#0f141b; --card:#171e28; --ink:#e7ecf2; --muted:#9aa6b6; --line:#29333f; --accent:#6ea3f2; --warn:#e6b450; --warnbg:#3a2c12; --good:#5cc98a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; display: grid; gap: 12px; }
h1 { margin: 0; font-size: 24px; } p { margin: 0; color: var(--muted); }
.row { display:grid; grid-template-columns: auto minmax(0,1fr) auto; gap: 8px 14px; align-items:center; padding:10px; background:var(--card); border:1px solid var(--line); border-radius:8px; }
.row.check { border-color: var(--warn); }
.pics { display:flex; gap:6px; } figure { margin:0; display:grid; justify-items:center; gap:2px; }
figcaption { font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
img { width:84px; height:84px; object-fit:contain; border-radius:6px; background:#fff; display:block; }
.id { font: 13px ui-monospace, Menlo, Consolas, monospace; color: var(--muted); }
.name { font-weight: 700; overflow-wrap: anywhere; }
.links { display:flex; gap:12px; flex-wrap:wrap; font-size:14px; margin-top:4px; } a { color: var(--accent); font-weight:700; text-decoration:none; } a:hover { text-decoration:underline; }
.side { display:grid; gap:6px; justify-items:end; text-align:right; }
.chip { font-size:12px; font-weight:700; padding:2px 8px; border-radius:999px; white-space:nowrap; background: color-mix(in srgb, var(--good) 15%, transparent); color: var(--good); }
.chip.warn { background: var(--warnbg); color: var(--warn); }
.price { font: 12px ui-monospace, Menlo, Consolas, monospace; }
.note { grid-column: 2 / -1; font-size: 13px; color: var(--muted); }
@media (max-width: 560px) { .row { grid-template-columns: minmax(0,1fr); } .side { justify-items:start; text-align:left; } .note { grid-column: 1; } img { width: 64px; height: 64px; } }
"""


def write_review_html(path, rows, title, intro, rb_colors=None):
    """A page showing each crop next to the matched part's picture, with BrickLink and Rebrickable links."""
    from html import escape as e
    cards = []
    for i, r in enumerate(rows, 1):
        kind = row_type(r)
        no = (r.get("bricklink_id") or r.get("part_id") or "").strip()
        blc = bl_colour(r.get("color_name"), rb_colors) if kind == "part" else None
        rbnum = r.get("rb_fig") if kind == "fig" and r.get("rb_fig") else (r.get("rb_part") or r.get("part_id") or "")
        pics = ""
        if r.get("crop_file"):
            src = _thumb_uri(r["crop_file"]) or _rel(r["crop_file"], path)
            pics += f'<figure><img src="{e(src)}" alt="Your item {i}"><figcaption>Yours</figcaption></figure>'
        if r.get("img_url"):
            pics += f'<figure><img src="{e(r["img_url"])}" alt="" loading="lazy"><figcaption>Match</figcaption></figure>'
        links = ""
        if no:
            links = (f'<a href="{e(bl_url(kind, no, blc, r.get("part_name")))}" target="_blank" rel="noopener">BrickLink ↗</a>'
                     f'<a href="{e(rb_url(kind, rbnum, r.get("part_name")))}" target="_blank" rel="noopener">Rebrickable ↗</a>')
        what = "Minifigure" if kind == "fig" else (r.get("color_name") or "Colour unknown") if kind == "part" else kind.title()
        score = r.get("match_score")
        chip = (f'<span class="chip warn">Check{f" · {float(score):.0%}" if score else ""}</span>' if r.get("needs_review")
                else f'<span class="chip">{float(score):.0%}</span>' if score else "")
        price = f'<span class="price">{e(money(r.get("price_each"), r.get("currency")))} each</span>' if r.get("price_each") else ""
        notes = "; ".join(x for x in (r.get("note"), f"Other guesses: {r['alternatives']}" if r.get("alternatives") else "") if x)
        cards.append(f'''<div class="row{" check" if r.get("needs_review") else ""}">
  <div class="pics">{pics}</div>
  <div><div class="id">#{i} · {e(r.get("part_id", ""))} · {e(what)} · qty {e(str(r.get("quantity", 1)))}</div>
    <div class="name">{e(r.get("part_name", ""))}</div><div class="links">{links}</div></div>
  <div class="side">{chip}{price}</div>
  {f'<div class="note">{e(notes)}</div>' if notes else ""}
</div>''')
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title><style>{PAGE_CSS}</style></head><body><main>
<h1>{e(title)}</h1><p>{e(intro)}</p>
{"".join(cards)}
</main></body></html>"""
    Path(path).write_text(html, encoding="utf-8")
    return path


def open_page(path, args):
    if not getattr(args, "no_open", False):
        import webbrowser
        webbrowser.open(Path(path).resolve().as_uri())


def add_prices(rows, args, rb_colors):
    """Fill price_each and currency on rows, using the BrickLink price guide."""
    if not bl_ready(explain=True):
        return 0
    n = 0
    for i, r in enumerate(rows, 1):
        print(f"\rGetting BrickLink prices ({i} of {len(rows)})...", end="", flush=True)
        p, why = price_row(r, args.condition, args.guide, args.currency, rb_colors)
        if p:
            r["price_each"], r["currency"] = f"{p['each']:.4f}", p.get("currency") or ""
            n += 1
        elif "BrickLink said" in why or "Couldn't reach" in why:
            print(f"\n{why}", file=sys.stderr)
            break
    print()
    return n


def cmd_value(args):
    if not bl_ready(explain=True):
        sys.exit(1)
    rows = read_inventory(args.inventory)
    if not rows:
        sys.exit(f"{args.inventory} is empty or missing. Run scan first.")
    key = args.key or os.environ.get("REBRICKABLE_API_KEY")
    cache_path = Path(args.inventory).resolve().with_name(".brick_sorter_cache.json")
    cache = load_cache(cache_path)
    rb_colors = rb_colours(key, cache)
    save_cache(cache_path, cache)
    priced = add_prices(rows, args, rb_colors)
    write_inventory(args.inventory, rows)
    total, cur, lines = 0.0, "", []
    for r in rows:
        if r.get("price_each"):
            v = float(r["price_each"]) * _qty(r)
            total += v
            cur = r.get("currency") or cur
            lines.append((v, r))
    lines.sort(key=lambda x: -x[0])
    cond = "new" if args.condition == "N" else "used"
    print(f"\nPriced {priced} of {len(rows)} rows. Estimated {cond} value: {money(total, cur)}")
    if priced < len(rows):
        print("Rows without a price have no colour, no BrickLink number, or no recent sales.")
    for v, r in lines[:10]:
        print(f"  {money(v, cur):>12}  {_qty(r):>3} x {r.get('part_id', ''):12} {r.get('color_name', '')[:18]:18} {r.get('part_name', '')[:40]}")
    out = args.out or "inventory_value.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "part_id", "part_name", "color_name", "quantity", "price_each", "value", "currency", "bricklink"])
        for v, r in lines:
            kind = row_type(r)
            w.writerow([kind, r.get("part_id"), r.get("part_name"), r.get("color_name"), _qty(r), r["price_each"], f"{v:.2f}",
                        r.get("currency"), bl_url(kind, r.get("bricklink_id") or r.get("part_id"),
                                                  bl_colour(r.get("color_name"), rb_colors) if kind == "part" else None)])
    print(f"\nUpdated price_each in {args.inventory} and saved {out}, most valuable first.")


# --------------------------------------------------------------------------
# Name the figure: which minifigure do these loose parts come from?
# --------------------------------------------------------------------------

def _same_num(a, b):
    a, b = str(a or "").lower(), str(b or "").lower()
    return bool(a) and (a == b or loose_part(a) == loose_part(b))


def name_fig_bricklink(parts):
    """BrickLink lists every minifigure that contains a part in a colour."""
    tally = {}
    for i, p in enumerate(parts):
        print(f"\rAsking BrickLink which minifigures have each part ({i + 1} of {len(parts)})...", end="", flush=True)
        for s in bricklink.supersets("part", p["no"], p.get("blc")):
            if s.get("type") != "MINIFIG":
                continue
            c = tally.setdefault(s["no"], {"bl": s["no"], "rb": "", "name": s.get("name"), "hits": set()})
            c["hits"].add(i)
    print()
    top = sorted(tally.values(), key=lambda c: -len(c["hits"]))[:8]
    for c in top:
        try:
            c["total"] = sum(1 for x in bricklink.subsets("fig", c["bl"]) if x.get("type") == "PART")
        except bricklink.BrickLinkError:
            c["total"] = None
        c["img"] = f"https://img.bricklink.com/ItemImage/MN/0/{c['bl']}.png"
    return top


def name_fig_rebrickable(parts, key, cache, rb_colors):
    """Start from the rarest part, find the figures in sets that have it, and check each one's parts."""
    lookup = build_color_lookup(rb_colors or [])
    for k in ("set_figs", "fig_parts", "fig_info"):
        cache.setdefault(k, {})
    for i, p in enumerate(parts, 1):
        print(f"\rLooking up the parts on Rebrickable ({i} of {len(parts)})...", end="", flush=True)
        p["rb"] = p.get("rb") or resolve_part({"part_id": p["id"], "bricklink_id": p["no"]}, key, cache)
        p["cid"] = lookup.get(norm_color(p["colour"])) if p.get("colour") else None
        p["n_sets"] = 0
        if p["rb"] and p["cid"] is not None:
            if p["rb"] not in cache["colors"]:
                res = rb_get_all(f"{REBRICKABLE_URL}/parts/{p['rb']}/colors/", key, missing_ok=True) or []
                cache["colors"][p["rb"]] = {str(c["color_id"]): c["num_sets"] for c in res}
            p["n_sets"] = cache["colors"][p["rb"]].get(str(p["cid"]), 0)
    print()
    seeds = sorted((p for p in parts if p["n_sets"]), key=lambda p: p["n_sets"])[:2]
    if not seeds:
        sys.exit("None of the parts could be found on Rebrickable in that colour. Check the colours, or add more parts.")
    figs = []
    for p in seeds:
        k = f"{p['rb']}|{p['cid']}"
        if k not in cache["pc_sets"]:
            res = rb_get_all(f"{REBRICKABLE_URL}/parts/{p['rb']}/colors/{p['cid']}/sets/", key, missing_ok=True) or []
            cache["pc_sets"][k] = [s["set_num"] for s in res]
        sets = cache["pc_sets"][k]
        figs += [s for s in sets if s.lower().startswith("fig-")]
        real = [s for s in sets if not s.lower().startswith("fig-")][:15]
        for i, s in enumerate(real, 1):
            if len(set(figs)) >= 40:
                break
            print(f"\rLooking at the minifigures in sets with {p['name']} ({i} of {len(real)})...", end="", flush=True)
            if s not in cache["set_figs"]:
                res = rb_get_all(f"{REBRICKABLE_URL}/sets/{s}/minifigs/", key, missing_ok=True) or []
                for f in res:
                    cache["fig_info"][f["set_num"]] = {"name": f.get("set_name"), "img": f.get("set_img_url")}
                cache["set_figs"][s] = [f["set_num"] for f in res]
            figs += cache["set_figs"][s]
        print()
    out = []
    figs = list(dict.fromkeys(figs))[:40]
    for i, fig in enumerate(figs, 1):
        print(f"\rComparing {len(figs)} possible figures ({i} of {len(figs)})...", end="", flush=True)
        if fig not in cache["fig_parts"]:
            res = rb_get_all(f"{REBRICKABLE_URL}/minifigs/{fig}/parts/", key, missing_ok=True) or []
            cache["fig_parts"][fig] = [[x["part"]["part_num"], x["color"]["id"],
                                        (x["part"].get("external_ids") or {}).get("BrickLink") or []] for x in res]
        if fig not in cache["fig_info"]:
            d = rb_get_all(f"{REBRICKABLE_URL}/minifigs/{fig}/", key, missing_ok=True) or {}
            cache["fig_info"][fig] = {"name": d.get("name", fig), "img": d.get("set_img_url")}
        fp = cache["fig_parts"][fig]
        hits = {k for k, p in enumerate(parts)
                if any((_same_num(num, p.get("rb")) or _same_num(num, p["id"]) or any(_same_num(b, p["no"]) for b in bl))
                       and (p.get("cid") is None or cid == p["cid"]) for num, cid, bl in fp)}
        if hits:
            info = cache["fig_info"][fig]
            out.append({"bl": "", "rb": fig, "name": info.get("name"), "img": info.get("img"), "hits": hits, "total": len(fp)})
    print()
    return out


def crop_photos(photos, args, crops_dir):
    """Detect and crop every item in the photos: [(photo name, crop path)]."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for photo in photos:
        img = cv2.imread(str(photo))
        if img is None:
            print(f"Can't read {photo} (HEIC photos need converting to JPEG first)", file=sys.stderr)
            continue
        boxes = (detect_parts_model(img, args.model, conf=args.conf, imgsz=args.imgsz) if args.model
                 else detect_parts(img, sensitivity=args.sensitivity, min_size=args.min_size))
        print(f"{photo.name}: found {len(boxes)} parts")
        for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
            crop = img[y0:y1, x0:x1]
            s = 640 / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            out = crops_dir / f"{photo.stem}_{i:03d}.jpg"
            cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jobs.append((photo.name, out))
    return jobs


def list_photos(paths):
    photos = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            photos += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.exists():
            photos.append(p)
        else:
            print(f"Not found: {p}", file=sys.stderr)
    return photos


def cmd_name_fig(args):
    key = args.key or os.environ.get("REBRICKABLE_API_KEY")
    use_bl = bl_ready()
    if not use_bl and not key:
        sys.exit("Naming a figure needs BrickLink keys (bricklink_keys.json) or a Rebrickable key (--key).")
    cache_path = Path(args.cache).resolve()
    cache = load_cache(cache_path)
    rb_colors = rb_colours(key, cache)
    parts, rows = [], []
    for spec in args.part or []:
        pid, _, colour = spec.partition(":")
        parts.append({"id": pid.strip(), "no": pid.strip(), "name": pid.strip(), "colour": colour.strip()})
        rows.append({"type": "part", "part_id": pid.strip(), "part_name": "", "color_name": colour.strip(), "quantity": 1})
    if args.photos:
        jobs = crop_photos(list_photos(args.photos), args, Path(args.crops))
        print(f"Identifying {len(jobs)} parts with Brickognize...")
        session = requests.Session()
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda j: identify(j[1], session, "parts"), jobs))
        for (photo, crop), res in zip(jobs, results):
            row = to_row(res, photo, str(crop), 0.7)
            if row.get("type") != "part":
                print(f"  {Path(crop).name}: not recognised as a part, left out")
                continue
            rows.append(row)
            parts.append({"id": row["part_id"], "no": row.get("bricklink_id") or row["part_id"],
                          "name": row["part_name"] or row["part_id"], "colour": row.get("color_name", "")})
            print(f"  {Path(crop).name}: {row['part_id']} {row.get('color_name', '')} {row['part_name'][:50]}")
    if not parts:
        sys.exit("Give photos of the figure's loose parts, or parts with --part ID:Colour.")
    for p in parts:
        p["blc"] = bl_colour(p["colour"], rb_colors)
    global RB_PACE
    RB_PACE = 1.0
    try:
        cands = None
        if use_bl:
            try:
                cands = name_fig_bricklink(parts)
            except bricklink.BrickLinkError as e:
                if not key:
                    sys.exit(str(e))
                print(f"{e}\nTrying Rebrickable instead.")
        if cands is None:
            cands = name_fig_rebrickable(parts, key, cache, rb_colors)
    finally:
        RB_PACE = 0.0
        save_cache(cache_path, cache)
    for c in cands:
        c["matched"] = len(c["hits"])
    cands.sort(key=lambda c: (-c["matched"], c.get("total") or 99))
    best = cands[0]["matched"] if cands else 0
    cands = [c for c in cands if c["matched"] >= max(1, best - 1)][:args.top]
    if not cands:
        sys.exit("No minifigure has these parts. Check each part and colour, or leave out parts from other figures.")
    print(f"\nBest matches for {len(parts)} parts:")
    figrows = []
    for i, c in enumerate(cands, 1):
        no = c["bl"] or c["rb"]
        row = {"type": "fig", "part_id": no, "part_name": c.get("name") or no, "bricklink_id": c["bl"], "rb_fig": c["rb"],
               "img_url": c.get("img") or "", "quantity": 1,
               "note": f"has {c['matched']} of your {len(parts)} parts" + (f"; {c['total']} parts in the figure" if c.get("total") else "")}
        if use_bl and args.prices and c["bl"]:
            p, _ = price_row(row, args.condition, args.guide, args.currency)
            if p:
                row["price_each"], row["currency"] = f"{p['each']:.4f}", p.get("currency") or ""
        figrows.append(row)
        print(f"{i:>2}. {no:<12} {row['part_name']}  ({row['note']})"
              + (f"  ~{money(row['price_each'], row['currency'])}" if row.get("price_each") else ""))
        print(f"    {bl_url('fig', no, name=row['part_name'])}")
    page = Path(args.out)
    write_review_html(page, figrows + rows, "Which minifigure is this?",
                      f"The best matches for your {len(parts)} parts come first, then the parts themselves. "
                      "Open BrickLink or Rebrickable to compare the pictures.", rb_colors)
    print(f"\nSaved {page}. To add the figure to your inventory, add a row with type fig and part_id {cands[0]['bl'] or cands[0]['rb']}"
          + (f" (rb_fig {cands[0]['rb']})" if cands[0]["rb"] else "") + ".")
    open_page(page, args)


# --------------------------------------------------------------------------
# Rebrickable
# --------------------------------------------------------------------------

RB_PACE = 0.0          # seconds between Rebrickable calls; set while suggesting sets
_rb_last = [0.0]


def rb_get_all(url, key, params=None, missing_ok=False):
    headers = {"Authorization": f"key {key}"}
    params = dict(params or {}, page_size=1000)
    out = []
    while url:
        if RB_PACE:
            wait = _rb_last[0] + RB_PACE - time.time()
            if wait > 0:
                time.sleep(wait)
            _rb_last[0] = time.time()
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code == 404 and missing_ok:
            return None
        if r.status_code == 401:
            sys.exit("Rebrickable rejected the API key. Check it at rebrickable.com > Settings > API.")
        if r.status_code == 404:
            sys.exit(f"Rebrickable doesn't know {url.rstrip('/').split('/')[-1] if '/sets/' in url else 'that'}."
                     " Check the set number; try adding the version, e.g. 10696-1.")
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        data = r.json()
        if "results" not in data:
            return data
        out += data["results"]
        url, params = data.get("next"), None
    return out


def norm_set(num):
    num = num.strip()
    return num if "-" in num else f"{num}-1"


def fetch_set(set_num, key, minifig_parts=False, figs=True):
    """Set info, its parts and (optionally) its minifigures as whole figures."""
    info = rb_get_all(f"{REBRICKABLE_URL}/sets/{set_num}/", key)
    parts = rb_get_all(f"{REBRICKABLE_URL}/sets/{set_num}/parts/", key,
                       {"inc_part_details": 1, "inc_minifig_parts": 1 if minifig_parts else 0})
    fig_list = rb_get_all(f"{REBRICKABLE_URL}/sets/{set_num}/minifigs/", key) if figs else []
    return info, parts, fig_list


def merge_lines(parts, set_num, include_spares=False):
    """One line per (part, colour) with the quantity the set needs."""
    merged = {}
    for p in parts:
        if p.get("is_spare") and not include_spares:
            continue
        part = p["part"]
        k = (part["part_num"], p["color"]["id"])
        if k in merged:
            merged[k]["quantity"] += p["quantity"]
            continue
        merged[k] = {
            "set_num": set_num,
            "part_num": part["part_num"],
            "bl_ids": (part.get("external_ids") or {}).get("BrickLink") or [],
            "name": part["name"],
            "color_id": p["color"]["id"],
            "color_name": p["color"]["name"],
            "quantity": p["quantity"],
            "img": part.get("part_img_url"),
        }
    return list(merged.values())


def fig_lines(fig_list, set_num):
    """One line per minifigure the set includes."""
    merged = {}
    for f in fig_list:
        num = f["set_num"]
        if num in merged:
            merged[num]["quantity"] += f.get("quantity", 1)
            continue
        merged[num] = {"set_num": set_num, "fig_num": num, "name": f.get("set_name") or f.get("name", ""),
                       "quantity": f.get("quantity", 1), "img": f.get("set_img_url")}
    return list(merged.values())


def norm_color(name):
    n = (name or "").lower().replace("grey", "gray")
    return re.sub(r"[^a-z0-9]", "", n)


def loose_part(pid):
    """3001b -> 3001, 3626cpb12 stays as is: only strip a single trailing mould letter."""
    m = re.fullmatch(r"(\d+)[a-z]", (pid or "").lower())
    return m.group(1) if m else (pid or "").lower()


def build_color_lookup(colors):
    """Map normalised colour names (Rebrickable and BrickLink spellings) to Rebrickable ids."""
    lookup = {}
    for c in colors:
        lookup[norm_color(c["name"])] = c["id"]
        bl = (c.get("external_ids") or {}).get("BrickLink") or {}
        for names in bl.get("ext_descrs") or []:
            for n in names:
                lookup.setdefault(norm_color(n), c["id"])
    return lookup


def _rgb(hexstr):
    try:
        return tuple(int(hexstr[i:i + 2], 16) for i in (0, 2, 4))
    except (TypeError, ValueError):
        return None


class SetCatalogue:
    """Parts lists and minifigures of the sets you expect in the pile.

    A Brickognize guess that is in one of your sets gets a bonus, so a close
    second guess that fits your sets beats a top guess that doesn't. Colours
    are limited to the colours that part comes in within your sets. Scanned
    minifigures are linked to your sets' figures by name.
    """
    BOOST = 0.2

    def __init__(self, sets, colors):
        self.sets = sets                                   # [(info, lines, figs)]
        self.colors = colors
        self.lookup = build_color_lookup(colors)
        self.by_id = {c["id"]: c for c in colors}
        self.index, self.loose = {}, {}
        self.figs = {}
        for info, lines, figs in sets:
            for l in lines:
                entry = self.index.get(l["part_num"].lower())
                if entry is None:
                    entry = {"part_num": l["part_num"], "name": l["name"], "bl_ids": l["bl_ids"],
                             "colors": defaultdict(int), "sets": set()}
                    for pid in [l["part_num"], *l["bl_ids"]]:
                        self.index.setdefault(pid.lower(), entry)
                        self.loose.setdefault(loose_part(pid), entry)
                entry["colors"][l["color_id"]] += l["quantity"]
                entry["sets"].add(info["set_num"])
            for f in figs:
                fig = self.figs.setdefault(f["fig_num"], dict(f, sets=set()))
                fig["sets"].add(info["set_num"])

    @classmethod
    def fetch(cls, set_nums, key):
        sets = []
        for num in set_nums:
            # Minifigure parts are included here so loose heads, torsos and
            # legs are steered too; spares count as evidence of what's in the pile.
            info, parts, figs = fetch_set(norm_set(num), key, minifig_parts=True, figs=True)
            sets.append((info, merge_lines(parts, info["set_num"], include_spares=True),
                         fig_lines(figs, info["set_num"])))
        colors = rb_get_all(f"{REBRICKABLE_URL}/colors/", key)
        return cls(sets, colors)

    def find(self, item):
        ids = [str(item.get("id", "")).lower(), str(bricklink_id(item)).lower()]
        for pid in ids:
            if pid in self.index:
                return self.index[pid]
        for pid in ids:
            if loose_part(pid) in self.loose:
                return self.loose[loose_part(pid)]
        return None

    def find_fig(self, item):
        """Best expected figure for a Brickognize minifig guess: (fig, similarity, how)."""
        ids = {str(item.get("id", "")).lower(), str(bricklink_id(item)).lower()}
        for num, fig in self.figs.items():
            if num.lower() in ids:
                return fig, 1.0, "same ID"
        best, best_sim = None, 0.0
        for fig in self.figs.values():
            sim = name_similarity(item.get("name"), fig["name"])
            if sim > best_sim:
                best, best_sim = fig, sim
        how = f"name match {best_sim:.0%}"
        return (best, best_sim, how) if best_sim >= FIG_LINK_AT else (None, best_sim, how)

    def choose(self, items, colors, mode="parts"):
        best = None
        for rank, it in enumerate(items):
            kind = item_type(it, mode)
            entry = fig = None
            sim, how = 0.0, ""
            if kind == "part":
                entry = self.find(it)
            elif kind == "fig":
                fig, sim, how = self.find_fig(it)
            adj = float(it.get("score", 0)) + (self.BOOST if (entry or fig) else 0)
            if best is None or adj > best[0]:
                best = (adj, rank, it, kind, entry, fig, sim, how)
        _, rank, it, kind, entry, fig, sim, how = best
        pick = {"item": it, "kind": kind, "entry": entry, "fig": fig, "fig_sim": sim,
                "color_name": "", "color_id": "", "notes": []}
        notes = pick["notes"]
        if rank > 0 and (entry or fig):
            top = items[0]
            notes.append(f"picked guess #{rank + 1} because it is in your sets "
                         f"(#1 was {top.get('id')} {top.get('name')})")

        if kind == "fig":
            if fig:
                notes.append(f"linked to {fig['fig_num']} {fig['name']} in your sets ({how})")
            else:
                notes.append("no figure in your expected sets has a similar name")
            return pick
        if kind != "part":
            return pick
        if not entry:
            col = colors[0] if colors else {}
            pick.update(color_name=col.get("name", ""), color_id=col.get("id", ""))
            notes.append("not in your expected sets")
            return pick

        allowed = entry["colors"]
        sets = ", ".join(sorted(entry.get("sets") or []))
        if colors:
            # Keep Brickognize's most likely colour; if your sets only have this part in other
            # colours, say which and flag it for checking.
            top = max(colors, key=lambda c: float(c.get("score", 0)))
            guess = self.lookup.get(norm_color(top.get("name")))
            pick.update(color_name=top.get("name", ""), color_id=guess if guess is not None else top.get("id", ""))
            if guess in allowed:
                notes.append(f"in your set{'s' if ',' in sets else ''} {sets}" if sets else "")
            else:
                alt = " or ".join(self.by_id[a]["name"] for a in list(allowed)[:3] if a in self.by_id)
                notes.append(f"this part is in {sets} in {alt}, not {top.get('name')}: check its colour")
                pick["colour_mismatch"] = True
            notes[:] = [n for n in notes if n]
            return pick
        cid = None
        if cid is None:
            if len(allowed) == 1:
                cid = next(iter(allowed))
                notes.append(f"colour set to {self.by_id[cid]['name']}, the only colour "
                             f"this part has in your sets")
            else:
                top_rb = self.lookup.get(norm_color(colors[0].get("name"))) if colors else None
                target = _rgb(self.by_id[top_rb]["rgb"]) if top_rb in self.by_id else None
                if target:
                    cid = min(allowed, key=lambda a: sum((x - y) ** 2 for x, y in
                                                         zip(_rgb(self.by_id[a]["rgb"]) or (0, 0, 0), target)))
                    notes.append(f"colour set to {self.by_id[cid]['name']}, the closest colour "
                                 f"this part has in your sets")
                else:
                    cid = max(allowed, key=allowed.get)
                    notes.append(f"colour guessed as {self.by_id[cid]['name']}, the most common "
                                 f"colour of this part in your sets")
        pick.update(color_name=self.by_id[cid]["name"], color_id=cid)
        return pick


# --------------------------------------------------------------------------
# Comparison against one or more sets
# --------------------------------------------------------------------------

def _qty(row):
    try:
        return int(float(row.get("quantity") or 1))
    except ValueError:
        return 1


def compare(set_lines, inventory, color_lookup):
    """Allocate inventory parts to set lines. Returns (lines, leftover rows).

    Leftover rows keep the inventory columns, so they can be passed on to the
    next set; that way a piece is never counted for two sets. Minifigure rows
    pass straight through.
    """
    pool = defaultdict(int)
    rows = {}
    others = []
    for row in inventory:
        if row_type(row) != "part":
            others.append(row)
            continue
        pid = (row.get("part_id") or "").strip()
        rb = (row.get("rb_part") or "").strip()
        if not pid and not rb:
            continue
        cid = color_lookup.get(norm_color(row.get("color_name"))) if row.get("color_name") else None
        key = (pid.lower(), (row.get("bricklink_id") or "").strip().lower(), rb.lower(), cid)
        pool[key] += _qty(row)
        rows.setdefault(key, row)

    def part_matches(key, line, loose=False):
        pid, blid, rb, _ = key
        ids = {line["part_num"].lower(), *[b.lower() for b in line["bl_ids"]]}
        if (rb and rb == line["part_num"].lower()) or pid in ids or (blid and blid in ids):
            return True
        if loose:
            lids = {loose_part(i) for i in ids}
            return any(x and loose_part(x) in lids for x in (pid, blid, rb))
        return False

    results = []
    for line in set_lines:
        need = line["quantity"]
        got, how = 0, ""
        # Pass 1: exact part + exact colour. Pass 2: exact part, colour unknown.
        # Pass 3: same part in a different mould variant (3001 vs 3001b), same colour.
        passes = [
            (False, lambda c: c == line["color_id"], "exact"),
            (False, lambda c: c is None, "colour unknown"),
            (True, lambda c: c == line["color_id"], "similar mould"),
        ]
        for loose, color_ok, label in passes:
            for key in list(pool):
                if got >= need:
                    break
                if pool[key] <= 0 or not color_ok(key[3]) or not part_matches(key, line, loose):
                    continue
                take = min(pool[key], need - got)
                pool[key] -= take
                got += take
                how = how or label
        results.append(dict(line, have=got, missing=need - got, match=how))

    leftovers = list(others)
    for key, qty in pool.items():
        if qty > 0:
            leftovers.append(dict(rows[key], quantity=qty,
                                  in_set_other_colour=any(part_matches(key, l, True) for l in set_lines)))
    return results, leftovers


def compare_figs(lines, inventory):
    """Allocate scanned minifigures to a set's figures.

    A scanned figure counts when it was linked to that figure (rb_fig, set by
    --sets or by editing the CSV), has the same ID, or, failing those, has a
    similar name. Name matches are marked so they can be checked.
    """
    pool, others = [], []
    for row in inventory:
        if row_type(row) == "fig":
            pool.append([row, _qty(row)])
        else:
            others.append(row)

    def linked(row, line):
        num = line["fig_num"].lower()
        return (row.get("rb_fig") or "").strip().lower() == num or (row.get("part_id") or "").strip().lower() == num

    results = []
    for line in lines:
        need, got, how = line["quantity"], 0, ""
        for method in ("id", "name"):
            for item in pool:
                if got >= need:
                    break
                row, left = item
                if left <= 0:
                    continue
                if method == "id":
                    ok = linked(row, line)
                else:
                    ok = not (row.get("rb_fig") or "").strip() and \
                        name_similarity(row.get("part_name"), line["name"]) >= FIG_SURE_AT
                if ok:
                    take = min(left, need - got)
                    item[1] -= take
                    got += take
                    how = how or ("linked" if method == "id" else "by name, check")
        results.append(dict(line, have=got, missing=need - got, match=how))
    leftovers = others + [dict(row, quantity=left, in_set_other_colour=False) for row, left in pool if left > 0]
    return results, leftovers


def write_report(path, results, leftovers, figs_whole=True):
    """results: [(set info, part lines, figure lines)] in the order pieces were shared out."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="1F3A5F")
    head_font = Font(bold=True, color="FFFFFF")
    miss_fill = PatternFill("solid", fgColor="FCE4E4")
    part_fill = PatternFill("solid", fgColor="FFF4CC")

    def table(ws, headers, data, fill_rule=None, start=1):
        for c, h in enumerate(headers, 1):
            cell = ws.cell(start, c, h)
            cell.font, cell.fill = head_font, head_fill
        for r, row in enumerate(data, start + 1):
            for c, v in enumerate(row, 1):
                ws.cell(r, c, v)
            f = fill_rule(row) if fill_rule else None
            if f:
                for c in range(1, len(headers) + 1):
                    ws.cell(r, c).fill = f
        for c, h in enumerate(headers, 1):
            width = max([len(str(h))] + [len(str(row[c - 1])) for row in data if not str(row[c - 1]).startswith("=")]) + 2
            ws.column_dimensions[get_column_letter(c)].width = min(width, 60)
        ws.freeze_panes = ws.cell(start + 1, 1)
        ws.auto_filter.ref = f"A{start}:{get_column_letter(len(headers))}{start + max(1, len(data))}"

    ws = wb.active
    ws.title = "Summary"
    summary = []
    for i, (info, lines, figs) in enumerate(results, 2):
        summary.append([info.get("set_num"), info.get("name"), info.get("year"),
                        sum(l["quantity"] for l in lines), sum(l["have"] for l in lines),
                        f"=D{i}-E{i}", f"=IF(D{i}=0,0,E{i}/D{i})",
                        sum(1 for l in lines if l["missing"] > 0),
                        sum(f["quantity"] for f in figs), sum(f["have"] for f in figs),
                        info.get("set_url", "")])
    table(ws, ["Set", "Name", "Year", "Pieces needed", "You have", "Missing", "Complete",
               "Different parts missing", "Minifigures needed", "Minifigures you have", "Rebrickable"], summary)
    for i in range(2, len(summary) + 2):
        ws.cell(i, 7).number_format = "0%"
    notes = ["Spare parts are left out."]
    if figs_whole:
        notes.append("Minifigures are counted as whole figures; their parts aren't in the piece counts.")
    else:
        notes.append("Minifigures are counted as their separate parts.")
    if len(results) > 1:
        notes.append("Pieces are shared out in the order the sets are listed, so none is counted twice.")
    row = len(summary) + 3
    for n in notes + [f"Items not used by any set: {sum(_qty(r) for r in leftovers)}"]:
        ws.cell(row, 1, n)
        row += 1

    def bl_link(l):
        bl = l["bl_ids"][0] if l["bl_ids"] else l["part_num"]
        return f"https://www.bricklink.com/v2/catalog/catalogitem.page?P={bl}"

    all_lines = [l for _, lines, _ in results for l in lines]
    missing = sorted((l for l in all_lines if l["missing"] > 0),
                     key=lambda l: (l["set_num"], -l["missing"], l["name"]))
    table(wb.create_sheet("Missing"),
          ["Set", "Part", "Name", "Colour", "Need", "Have", "Missing", "BrickLink", "Image"],
          [[l["set_num"], l["part_num"], l["name"], l["color_name"], l["quantity"], l["have"],
            l["missing"], bl_link(l), l["img"] or ""] for l in missing])
    table(wb.create_sheet("All set parts"),
          ["Set", "Part", "Name", "Colour", "Need", "Have", "Missing", "Matched by"],
          [[l["set_num"], l["part_num"], l["name"], l["color_name"], l["quantity"], l["have"],
            l["missing"], l["match"]]
           for l in sorted(all_lines, key=lambda l: (l["set_num"], l["missing"] == 0, l["name"]))],
          fill_rule=lambda r: miss_fill if r[5] == 0 else (part_fill if r[6] > 0 else None))
    all_figs = [f for _, _, figs in results for f in figs]
    if all_figs:
        table(wb.create_sheet("Minifigures"),
              ["Set", "Figure", "Name", "Need", "Have", "Missing", "Matched by", "Rebrickable", "Image"],
              [[f["set_num"], f["fig_num"], f["name"], f["quantity"], f["have"], f["missing"], f["match"],
                f"https://rebrickable.com/minifigs/{f['fig_num']}/", f["img"] or ""]
               for f in sorted(all_figs, key=lambda f: (f["set_num"], f["missing"] == 0, f["name"]))],
              fill_rule=lambda r: miss_fill if r[4] == 0 else (part_fill if "check" in (r[6] or "") else None))
    table(wb.create_sheet("Not in any set" if len(results) > 1 else "Not in set"),
          ["Type", "ID", "Name", "Colour", "Quantity", "Note"],
          [[row_type(r), r.get("part_id") or r.get("rb_part"), r.get("part_name"), r.get("color_name"), _qty(r),
            "in a set in a different colour (check the colour)" if r.get("in_set_other_colour") else ""]
           for r in sorted(leftovers, key=lambda r: (row_type(r), r.get("part_name") or ""))])
    wb.save(path)


def cmd_compare(args):
    key = args.key or os.environ.get("REBRICKABLE_API_KEY")
    if not key:
        sys.exit("Needs a Rebrickable API key: --key or REBRICKABLE_API_KEY.")
    remaining = read_inventory(args.inventory)
    if not remaining:
        sys.exit(f"{args.inventory} is empty or missing. Run scan first.")

    whole = args.minifigs == "whole"
    colors = rb_get_all(f"{REBRICKABLE_URL}/colors/", key)
    lookup = build_color_lookup(colors)
    results = []
    for num in args.sets:
        set_num = norm_set(num)
        print(f"Fetching {set_num} from Rebrickable...")
        info, parts, figs = fetch_set(set_num, key, minifig_parts=not whole, figs=whole)
        lines, remaining = compare(merge_lines(parts, info["set_num"], args.include_spares), remaining, lookup)
        flines, remaining = compare_figs(fig_lines(figs, info["set_num"]), remaining)
        results.append((info, lines, flines))

    nums = [norm_set(n) for n in args.sets]
    out = args.out or (f"{nums[0]}_check.xlsx" if len(nums) == 1 else f"{len(nums)}_sets_check.xlsx")
    write_report(out, results, remaining, whole)

    for info, lines, figs in results:
        need = sum(l["quantity"] for l in lines)
        have = sum(l["have"] for l in lines)
        pct = 100 * have / need if need else 0
        msg = (f"{info['set_num']} {info.get('name')}: {have} of {need} pieces ({pct:.0f}%), "
               f"{need - have} missing across {sum(1 for l in lines if l['missing'])} different parts.")
        if figs:
            fneed, fhave = sum(f["quantity"] for f in figs), sum(f["have"] for f in figs)
            by_name = sum(1 for f in figs if "check" in f["match"])
            msg += f" Minifigures: {fhave} of {fneed}" + (f" ({by_name} matched by name, check them)" if by_name else "") + "."
        print(msg)
    if len(results) > 1:
        print("Pieces were shared out in the order you listed the sets; none is counted twice.")
    print(f"{sum(_qty(r) for r in remaining)} items aren't used by any of these sets.")
    print(f"Report written to {out}")


# --------------------------------------------------------------------------
# Suggesting sets from what you have
# --------------------------------------------------------------------------

TOTAL_SETS = 25000
DEFAULT_WEIGHT = 1.5


def rarity(n_sets):
    """A part in few sets is strong evidence; one in thousands is weak."""
    return math.log(1 + TOTAL_SETS / max(1, n_sets))


def tally_candidates(evidence):
    """evidence: [{label, weight, sets}] -> [{set_num, score, clues}] best first."""
    by = {}
    for ev in evidence:
        for sn in set(ev["sets"]):
            c = by.setdefault(sn, {"set_num": sn, "score": 0.0, "clues": []})
            c["score"] += ev["weight"]
            c["clues"].append(ev["label"])
    return sorted(by.values(), key=lambda c: (-c["score"], c["set_num"]))


def score_candidate(lines, flines, inventory, lookup, weights, count_lines=None):
    """Check one candidate set against the whole inventory.

    count_lines: the set's parts without minifigure parts, for "you have N%"
    (the same count compare uses).
    """
    parts, _ = compare(lines, inventory, lookup)
    figs, _ = compare_figs(flines, inventory)
    counted = compare(count_lines, inventory, lookup)[0] if count_lines is not None else parts
    have = sum(l["have"] for l in counted)
    need = sum(l["quantity"] for l in counted)
    score, kinds, clues = 0.0, 0, []
    for l in parts:
        if not l["have"]:
            continue
        kinds += 1
        w = weights.get(f"{l['part_num'].lower()}|{l['color_id']}")
        score += w[0] if w else DEFAULT_WEIGHT
        if w:
            clues.append(w)
    for f in figs:          # whole figures add evidence; their parts are already in the piece counts
        if not f["have"]:
            continue
        kinds += 1
        w = weights.get(f"fig|{f['fig_num'].lower()}") or (rarity(20), f["name"])
        score += w[0]
        clues.append(w)
    clues.sort(key=lambda w: -w[0])
    return {"score": score, "have": have, "need": need, "kinds": kinds, "clues": [c[1] for c in clues[:4]]}


def load_cache(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"resolve": {}, "colors": {}, "pc_sets": {}, "fig_sets": {}}


def save_cache(path, cache):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        pass


def resolve_part(row, key, cache):
    """Rebrickable part number for a Brickognize ID (usually the same)."""
    if row.get("rb_part"):
        return row["rb_part"]
    pid = row["part_id"].strip()
    if pid in cache["resolve"]:
        return cache["resolve"][pid] or None
    num = None
    direct = rb_get_all(f"{REBRICKABLE_URL}/parts/{pid}/", key, missing_ok=True)
    if direct and direct.get("part_num"):
        num = direct["part_num"]
    else:
        hits = rb_get_all(f"{REBRICKABLE_URL}/parts/", key,
                          {"bricklink_id": (row.get("bricklink_id") or pid).strip()}, missing_ok=True)
        if hits:
            num = hits[0]["part_num"]
    cache["resolve"][pid] = num or ""
    return num


def cmd_suggest(args):
    global RB_PACE
    key = args.key or os.environ.get("REBRICKABLE_API_KEY")
    if not key:
        sys.exit("Needs a Rebrickable API key: --key or REBRICKABLE_API_KEY.")
    inventory = read_inventory(args.inventory)
    part_rows = [r for r in inventory if row_type(r) == "part" and (r.get("part_id") or "").strip()]
    fig_rows = [r for r in inventory if row_type(r) == "fig" and (r.get("rb_fig") or "").strip()]
    if not part_rows and not fig_rows:
        sys.exit(f"{args.inventory} has no parts or linked minifigures. Run scan first.")
    cache_path = Path(args.inventory).resolve().with_name(".brick_sorter_cache.json")
    cache = load_cache(cache_path)
    RB_PACE = 1.0
    try:
        colors = rb_get_all(f"{REBRICKABLE_URL}/colors/", key)
        lookup = build_color_lookup(colors)
        names = {c["id"]: c["name"] for c in colors}

        distinct = {}
        for r in part_rows:
            d = distinct.setdefault(r["part_id"].strip().lower(), {"row": r, "colours": set()})
            cid = lookup.get(norm_color(r.get("color_name"))) if r.get("color_name") else None
            if cid is not None:
                d["colours"].add(cid)
        todo = [d for d in distinct.values() if d["colours"]][:args.max_parts]

        clues = []
        for i, d in enumerate(todo, 1):
            print(f"\rChecking how rare your parts are ({i} of {len(todo)})...", end="", flush=True)
            num = resolve_part(d["row"], key, cache)
            if not num:
                continue
            if num not in cache["colors"]:
                res = rb_get_all(f"{REBRICKABLE_URL}/parts/{num}/colors/", key, missing_ok=True) or []
                cache["colors"][num] = {str(c["color_id"]): c["num_sets"] for c in res}
            for cid in d["colours"]:
                n = cache["colors"][num].get(str(cid))
                if not n:
                    continue        # never appeared in that colour: probably a wrong colour guess
                clues.append({"num": num, "cid": cid, "n": n, "weight": rarity(n),
                              "label": f"{d['row'].get('part_name') or num} in {names.get(cid, cid)}"})
            if i % 10 == 0:
                save_cache(cache_path, cache)
        print()
        weights = {f"{c['num'].lower()}|{c['cid']}": (c["weight"], c["label"]) for c in clues}

        rare = sorted((c for c in clues if c["n"] <= args.max_common), key=lambda c: c["n"])[:args.clues]
        evidence = []
        for i, c in enumerate(rare, 1):
            print(f"\rFinding sets with your rarest parts ({i} of {len(rare)})...", end="", flush=True)
            k = f"{c['num']}|{c['cid']}"
            if k not in cache["pc_sets"]:
                res = rb_get_all(f"{REBRICKABLE_URL}/parts/{c['num']}/colors/{c['cid']}/sets/", key, missing_ok=True) or []
                cache["pc_sets"][k] = [s["set_num"] for s in res]
            evidence.append({"label": c["label"], "weight": c["weight"], "sets": cache["pc_sets"][k]})
        if rare:
            print()
        for fig in sorted({r["rb_fig"].strip() for r in fig_rows}):
            if fig not in cache["fig_sets"]:
                res = rb_get_all(f"{REBRICKABLE_URL}/minifigs/{fig}/sets/", key, missing_ok=True) or []
                cache["fig_sets"][fig] = [s["set_num"] for s in res]
            sets = cache["fig_sets"][fig]
            name = next((r.get("part_name") for r in fig_rows if r["rb_fig"].strip() == fig), fig)
            evidence.append({"label": f"Minifigure {name}", "weight": rarity(len(sets)), "sets": sets})
            weights[f"fig|{fig.lower()}"] = (rarity(len(sets)), f"Minifigure {name}")
        save_cache(cache_path, cache)
        if not evidence:
            sys.exit("None of your parts or minifigures narrowed it down: they're very common or unknown to "
                     "Rebrickable. Scan more parts, especially unusual or printed ones, and try again.")

        results = []
        top = tally_candidates(evidence)[:args.top]
        for i, cand in enumerate(top, 1):
            print(f"\rComparing the best matches ({i} of {len(top)})...", end="", flush=True)
            info = rb_get_all(f"{REBRICKABLE_URL}/sets/{cand['set_num']}/", key, missing_ok=True)
            if not info:
                continue
            parts = rb_get_all(f"{REBRICKABLE_URL}/sets/{cand['set_num']}/parts/", key,
                               {"inc_part_details": 1, "inc_minifig_parts": 1})
            bricks = rb_get_all(f"{REBRICKABLE_URL}/sets/{cand['set_num']}/parts/", key,
                                {"inc_part_details": 1, "inc_minifig_parts": 0})
            figs = rb_get_all(f"{REBRICKABLE_URL}/sets/{cand['set_num']}/minifigs/", key) or []
            sc = score_candidate(merge_lines(parts, info["set_num"]), fig_lines(figs, info["set_num"]),
                                 inventory, lookup, weights, merge_lines(bricks, info["set_num"]))
            results.append(dict(sc, info=info))
        print()
    finally:
        save_cache(cache_path, cache)
        RB_PACE = 0.0

    results.sort(key=lambda r: (-r["score"], -(r["have"] / max(1, r["need"]))))
    print(f"\nChecked {len(todo)} kinds of part and {len({r['rb_fig'] for r in fig_rows})} linked minifigures.\n")
    out_rows = []
    for i, r in enumerate(results, 1):
        pct = 100 * r["have"] / r["need"] if r["need"] else 0
        info = r["info"]
        print(f"{i:>2}. {info['set_num']:<10} {info.get('name', '')} ({info.get('year')})")
        print(f"    You have {pct:.0f}% of it; {r['have']} of your pieces fit; {r['kinds']} kinds of part match.")
        if r["clues"]:
            print(f"    Strongest clues: {'; '.join(r['clues'])}")
        out_rows.append({"rank": i, "set_num": info["set_num"], "name": info.get("name"), "year": info.get("year"),
                         "percent_you_have": round(pct, 1), "pieces_that_fit": r["have"], "set_pieces": r["need"],
                         "kinds_matched": r["kinds"], "score": round(r["score"], 2),
                         "clues": "; ".join(r["clues"]), "rebrickable": info.get("set_url", "")})
    if out_rows:
        out = args.out or "suggested_sets.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nSaved to {out}. Check one with: python brick_sorter.py compare {out_rows[0]['set_num']}")
    else:
        print("No matching sets found.")


def add_price_args(p):
    p.add_argument("--condition", choices=["U", "N"], default="U", help="BrickLink prices for used (U, default) or new (N)")
    p.add_argument("--guide", choices=["sold", "stock"], default="sold",
                   help="sold: average of the last 6 months' sales (default); stock: items for sale now")
    p.add_argument("--currency", help="e.g. AUD (default: the currency in bricklink_keys.json, else BrickLink's)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="detect and identify parts or minifigures in photos")
    s.add_argument("photos", nargs="+", help="photo files or folders")
    s.add_argument("--mode", choices=["parts", "figs", "mixed"], default="parts",
                   help="what the photos show: parts (default), figs (assembled minifigures) "
                        "or mixed (Brickognize decides for each item)")
    s.add_argument("--sets", nargs="+", metavar="SET",
                   help="sets you think are in the pile, e.g. --sets 10696 6177; their parts lists "
                        "and minifigures steer the guesses (needs a Rebrickable key)")
    s.add_argument("--key", help="Rebrickable API key (for --sets)")
    s.add_argument("--inventory", default="inventory.csv")
    s.add_argument("--crops", default="crops", help="folder for the cropped items")
    s.add_argument("--sensitivity", type=float, default=1.0,
                   help="raise (e.g. 1.5) if parts are missed, lower (0.7) if shadows become parts")
    s.add_argument("--min-size", type=float, default=0.0003,
                   help="smallest part as a fraction of the photo area")
    s.add_argument("--model", help="find items with a model instead of the background method: your brick "
                                   "detector (lego-detector.pt), or the figure finder (yolo11n.pt or "
                                   "figure-finder.onnx) for minifigures, even on display shelves")
    s.add_argument("--conf", type=float, default=None,
                   help="with --model: lower finds more, higher ignores non-items (default 0.3 for the "
                        "brick detector, 0.1 for the figure finder)")
    s.add_argument("--tiles", type=int, default=None,
                   help="with --model: also search an N x N grid of zoomed-in tiles (default 3 for the "
                        "figure finder, 1 for the brick detector)")
    s.add_argument("--imgsz", type=int, default=960,
                   help="with --model: image size the model was trained at")
    s.add_argument("--review-below", type=float, default=0.7,
                   help="flag matches below this confidence for checking (without --sets)")
    s.add_argument("--debug", action="store_true", help="save each photo with numbered boxes drawn on it")
    s.add_argument("--detect-only", action="store_true", help="only crop, don't call Brickognize")
    s.add_argument("--prices", action="store_true", help="add BrickLink prices (needs bricklink_keys.json)")
    add_price_args(s)
    s.add_argument("--no-open", action="store_true", help="don't open scan_review.html in the browser")
    s.set_defaults(func=cmd_scan)

    c = sub.add_parser("compare", help="check your inventory against one or more sets")
    c.add_argument("sets", nargs="+", metavar="SET",
                   help="set numbers, e.g. 10696 or 10696-1; with several, pieces are shared out "
                        "in this order so none is counted twice")
    c.add_argument("--inventory", default="inventory.csv")
    c.add_argument("--key", help="Rebrickable API key")
    c.add_argument("--out", help="output .xlsx")
    c.add_argument("--include-spares", action="store_true")
    c.add_argument("--minifigs", choices=["whole", "parts"], default="whole",
                   help="whole (default): check minifigures as whole figures; parts: count their "
                        "heads, torsos, legs etc. as separate parts")
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("suggest", help="suggest which sets your inventory came from")
    g.add_argument("--inventory", default="inventory.csv")
    g.add_argument("--key", help="Rebrickable API key")
    g.add_argument("--top", type=int, default=8, help="how many candidate sets to check in full")
    g.add_argument("--clues", type=int, default=30, help="how many of your rarest part+colour combinations to look up")
    g.add_argument("--max-parts", type=int, default=120, help="most kinds of part to check for rarity")
    g.add_argument("--max-common", type=int, default=2000,
                   help="ignore part+colour combinations found in more sets than this")
    g.add_argument("--out", help="CSV of suggestions (default suggested_sets.csv)")
    g.set_defaults(func=cmd_suggest)

    v = sub.add_parser("value", help="price your inventory with BrickLink's price guide (needs BrickLink keys)")
    v.add_argument("--inventory", default="inventory.csv")
    v.add_argument("--key", help="Rebrickable API key (optional: better colour matching)")
    v.add_argument("--out", help="CSV of priced rows, most valuable first (default inventory_value.csv)")
    add_price_args(v)
    v.set_defaults(func=cmd_value)

    n = sub.add_parser("name-fig", help="work out which minifigure some loose parts come from")
    n.add_argument("photos", nargs="*", help="photos of one figure's loose parts (head, torso, legs...), spread out")
    n.add_argument("--part", action="append", metavar="ID:COLOUR",
                   help="a part you already know, e.g. --part 973pb1234c01:Red (repeat for each part)")
    n.add_argument("--key", help="Rebrickable API key (used when BrickLink keys aren't set up)")
    n.add_argument("--top", type=int, default=6, help="how many figures to list")
    n.add_argument("--out", default="name_fig.html", help="page with the matches (default name_fig.html)")
    n.add_argument("--cache", default=".brick_sorter_cache.json")
    n.add_argument("--crops", default="crops_fig", help="folder for the cropped parts")
    n.add_argument("--model", help="find parts with your brick detector instead of the background method")
    n.add_argument("--conf", type=float, default=None)
    n.add_argument("--imgsz", type=int, default=960)
    n.add_argument("--sensitivity", type=float, default=1.0)
    n.add_argument("--min-size", type=float, default=0.0003)
    n.add_argument("--prices", action="store_true", help="show each figure's BrickLink price")
    add_price_args(n)
    n.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
    n.set_defaults(func=cmd_name_fig)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
