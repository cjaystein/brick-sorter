#!/usr/bin/env python3
"""
Build rb-index.bin.gz: Rebrickable's whole LEGO database, packed small enough for
the web app to download once and search instantly (set suggestions, rare-part
flags and naming figures, with no waiting on Rebrickable's one-request-a-second
limit).

    python build_rb_index.py              # download the latest data and build
    python build_rb_index.py --data dir   # use CSV files you've already downloaded

Rebrickable publishes these files daily at https://rebrickable.com/downloads/
(free to use; please credit Rebrickable). Put rb-index.bin.gz next to
brick-sorter.html (in your GitHub repository for the website). Only standard
Python is needed.

File layout (all numbers little-endian):
  "RBIX" + uint32 header length + header JSON (UTF-8)
  then the arrays listed in header["arrays"], each a typed array of the given
  type and length, one after another.
Header: built (date), parts [part_num], part_names [name], part_cats [category id] (same order as parts),
  categories {id: name}, colors [[id, name, rgb]], themes {id: name},
  sets [[set_num, name, year, theme_id, num_parts, img]], figs [[fig_num, name, num_parts]].
Owners are the sets followed by the figures. For owner i, its parts are
  lines[line_start[i] : line_start[i+1]] in line_part (part index), line_color
  (colour id) and line_qty (quantity; +32768 when it's a spare). Its whole
  minifigures are fig_start/fig_idx/fig_qty the same way (sets only).
"""

import argparse
import csv
import gzip
import io
import json
import struct
import sys
import time
import urllib.request
from array import array
from pathlib import Path

FILES = ["sets", "minifigs", "inventories", "inventory_parts", "inventory_minifigs", "colors", "themes", "parts", "part_categories"]
URL = "https://cdn.rebrickable.com/media/downloads/{}.csv.gz"


def fetch(name, folder):
    path = Path(folder) / f"{name}.csv.gz"
    if path.exists() and time.time() - path.stat().st_mtime < 20 * 3600:
        return path
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(name), headers={"User-Agent": "BrickSorter-index-builder/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            gzip.decompress(data[:0] or data)          # check it's complete
            path.write_bytes(data)
            return path
        except Exception as e:                           # flaky connections: try again
            if attempt == 3:
                raise SystemExit(f"Couldn't download {name}: {e}")
            time.sleep(3 * (attempt + 1))


def rows(folder, name):
    with gzip.open(Path(folder) / f"{name}.csv.gz", "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def build(folder, out):
    sets = {r["set_num"]: r for r in rows(folder, "sets")}
    figs = {r["fig_num"]: r for r in rows(folder, "minifigs")}
    latest = {}                                          # newest inventory version of each set / figure
    for r in rows(folder, "inventories"):
        sn, v = r["set_num"], int(r["version"])
        if sn not in latest or v > latest[sn][1]:
            latest[sn] = (int(r["id"]), v)
    inv_owner = {iid: sn for sn, (iid, _) in latest.items()}

    set_list = sorted(s for s in sets if s in latest)
    fig_list = sorted(f for f in figs if f in latest)
    owners = set_list + fig_list
    owner_idx = {o: i for i, o in enumerate(owners)}
    fig_idx = {f: i for i, f in enumerate(fig_list)}

    lines = [[] for _ in owners]
    parts = {}
    for r in rows(folder, "inventory_parts"):
        o = inv_owner.get(int(r["inventory_id"]))
        if o is None or o not in owner_idx:
            continue
        p = r["part_num"]
        pi = parts.setdefault(p, len(parts))
        q = min(32767, int(r["quantity"])) + (32768 if r["is_spare"] == "True" else 0)
        lines[owner_idx[o]].append((pi, int(r["color_id"]), q))
    set_figs = [[] for _ in set_list]
    n_sets = len(set_list)
    for r in rows(folder, "inventory_minifigs"):
        o = inv_owner.get(int(r["inventory_id"]))
        if o is None or o not in owner_idx or owner_idx[o] >= n_sets or r["fig_num"] not in fig_idx:
            continue
        set_figs[owner_idx[o]].append((fig_idx[r["fig_num"]], min(255, int(r["quantity"]))))

    line_start, line_part, line_color, line_qty = array("I", [0]), array("I"), array("H"), array("H")
    for ls in lines:
        ls.sort()
        for pi, c, q in ls:
            line_part.append(pi); line_color.append(max(0, c) if c >= 0 else 65535); line_qty.append(q)
        line_start.append(len(line_part))
    fig_start, fig_i, fig_q = array("I", [0]), array("H"), array("B")
    for fl in set_figs:
        for fi, q in sorted(fl):
            fig_i.append(fi); fig_q.append(q)
        fig_start.append(len(fig_i))

    def img(sn, url):
        return "" if url == f"https://cdn.rebrickable.com/media/sets/{sn.lower()}.jpg" else (url or "-")
    names, cats = {}, {}
    if (Path(folder) / "parts.csv.gz").exists():
        for r in rows(folder, "parts"):
            names[r["part_num"]] = r["name"]
            cats[r["part_num"]] = int(r["part_cat_id"] or 0)
    categories = {r["id"]: r["name"] for r in rows(folder, "part_categories")} if (Path(folder) / "part_categories.csv.gz").exists() else {}
    header = {
        "built": time.strftime("%Y-%m-%d"),
        "source": "Rebrickable (rebrickable.com/downloads)",
        "parts": list(parts),
        "part_names": [names.get(p, "") for p in parts],
        "part_cats": [cats.get(p, 0) for p in parts],
        "categories": categories,
        "colors": [[int(r["id"]), r["name"], r["rgb"]] for r in rows(folder, "colors")],
        "themes": {r["id"]: r["name"] for r in rows(folder, "themes")},
        "sets": [[s, sets[s]["name"], int(sets[s]["year"] or 0), int(sets[s]["theme_id"] or 0), int(sets[s]["num_parts"] or 0),
                  img(s, sets[s]["img_url"])] for s in set_list],
        "figs": [[f, figs[f]["name"], int(figs[f]["num_parts"] or 0)] for f in fig_list],
    }
    arrays = [("line_start", line_start), ("line_part", line_part), ("line_color", line_color), ("line_qty", line_qty),
              ("fig_start", fig_start), ("fig_idx", fig_i), ("fig_qty", fig_q)]
    header["arrays"] = [[name, a.typecode, len(a)] for name, a in arrays]
    hj = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body = io.BytesIO()
    body.write(b"RBIX"); body.write(struct.pack("<I", len(hj))); body.write(hj)
    for _, a in arrays:
        if sys.byteorder != "little":
            a = array(a.typecode, a); a.byteswap()
        body.write(a.tobytes())
    raw = body.getvalue()
    Path(out).write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    meta = {"built": header["built"], "sets": len(set_list), "figs": len(fig_list), "parts": len(parts),
            "lines": len(line_part), "bytes": Path(out).stat().st_size}
    Path(out).with_name("rb-index.json").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder for the downloaded CSV files (default: rebrickable-data)")
    ap.add_argument("--out", default="rb-index.bin.gz")
    args = ap.parse_args()
    folder = Path(args.data or "rebrickable-data")
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        print(f"Getting {name}...")
        fetch(name, folder)
    print("Packing...")
    meta = build(folder, args.out)
    print(f"Saved {args.out}: {meta['sets']} sets, {meta['figs']} minifigures, {meta['lines']} parts-list lines, "
          f"{meta['bytes'] / 1e6:.1f} MB (built {meta['built']}).")


if __name__ == "__main__":
    main()
