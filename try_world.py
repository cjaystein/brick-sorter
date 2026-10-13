#!/usr/bin/env python3
"""
Try YOLO-World on a photo: it finds objects from a description in words,
with no training. Draws a numbered box around everything that matches and
saves <photo>_world.jpg next to the photo.

    python try_world.py shelf.jpg
    python try_world.py shelf.jpg --tiles 3 --conf 0.03
    python try_world.py shelf.jpg --words "lego minifigure" "toy figure" "person"
    python try_world.py shelf.jpg --tiles 3 --crops        # also save one image per figure
    python try_world.py shelf.jpg --model yolo11x.pt       # an ordinary model: finds them as "person"

--tiles N cuts the photo into an N x N grid of overlapping pieces and searches
each piece zoomed in, so small figures look bigger. Results are merged, so a
figure on a boundary is still counted once.

The first run downloads the model (about 140 MB) and its text model
(about 340 MB). Later runs reuse them.
"""

import argparse
import sys
from pathlib import Path


def tile_boxes(w, h, n, overlap=0.25):
    """n x n overlapping tiles covering a w x h image: (x0, y0, x1, y1)."""
    if n <= 1:
        return [(0, 0, w, h)]
    tw, th = w / (n - (n - 1) * overlap), h / (n - (n - 1) * overlap)
    sx, sy = tw * (1 - overlap), th * (1 - overlap)
    return [(round(i * sx), round(j * sy), round(min(w, i * sx + tw)), round(min(h, j * sy + th)))
            for j in range(n) for i in range(n)]


def detect(model, image, tiles, imgsz, conf, classes=None):
    """Run the model over the whole photo and each tile; return merged [x0, y0, x1, y1, score]."""
    import torch
    from torchvision.ops import nms

    h, w = image.shape[:2]
    regions = [(0, 0, w, h)] + (tile_boxes(w, h, tiles) if tiles > 1 else [])
    boxes, scores = [], []
    for x0, y0, x1, y1 in regions:
        r = model.predict(image[y0:y1, x0:x1], imgsz=imgsz, conf=conf, iou=0.5, max_det=500,
                          agnostic_nms=True, classes=classes, verbose=False)[0]
        for (bx0, by0, bx1, by1), s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()):
            # Skip boxes cut off by a tile edge (the whole figure is found in a neighbouring tile).
            if (x0, y0, x1, y1) != regions[0]:
                cut = (bx0 < 2 and x0 > 0) or (by0 < 2 and y0 > 0) or \
                      (bx1 > (x1 - x0) - 2 and x1 < w) or (by1 > (y1 - y0) - 2 and y1 < h)
                if cut:
                    continue
            if (bx1 - bx0) * (by1 - by0) > 0.08 * w * h:      # far bigger than one figure
                continue
            boxes.append([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0])
            scores.append(s)
    if not boxes:
        return []
    b, s = torch.tensor(boxes), torch.tensor(scores)
    keep = nms(b, s, 0.5).tolist()
    # Drop boxes that mostly sit inside a bigger kept box (e.g. a head found separately from its figure).
    kept = [boxes[i] + [scores[i]] for i in keep]
    area = lambda q: max(0, q[2] - q[0]) * max(0, q[3] - q[1])
    out = []
    for q in kept:
        inside = any(o is not q and area(o) > area(q) and
                     max(0, min(q[2], o[2]) - max(q[0], o[0])) * max(0, min(q[3], o[3]) - max(q[1], o[1])) > 0.8 * area(q)
                     for o in kept)
        if not inside:
            out.append(q)
    out.sort(key=lambda q: (round(q[1] / max(1, h * 0.08)), q[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", help="photo to search, e.g. shelf.jpg")
    ap.add_argument("--words", nargs="+", default=["lego minifigure"],
                    help='what to look for, e.g. --words "lego minifigure" "toy figure"')
    ap.add_argument("--conf", type=float, default=0.05,
                    help="lower (0.02) finds more, higher (0.15) ignores more non-figures")
    ap.add_argument("--tiles", type=int, default=2,
                    help="search an N x N grid of zoomed-in pieces as well as the whole photo (1 = off)")
    ap.add_argument("--imgsz", type=int, default=1280, help="larger helps with small figures, but is slower")
    ap.add_argument("--model", default="yolov8x-worldv2.pt",
                    help="yolov8s/m/l/x-worldv2.pt (finds what --words describes; x is the most accurate), "
                         "or an ordinary model such as yolo11x.pt, which looks for people")
    ap.add_argument("--crops", action="store_true", help="also save each figure as its own image")
    args = ap.parse_args()

    photo = Path(args.photo)
    if not photo.exists():
        sys.exit(f"Can't find {photo}. Put the photo in this folder or give its full path.")
    try:
        import cv2
        import torch
        from ultralytics import YOLO, YOLOWorld
    except ImportError:
        sys.exit("Ultralytics isn't installed here. Activate your venv first (venv\\Scripts\\activate).")

    classes = None
    if "world" in Path(args.model).name.lower():
        try:
            import clip  # noqa: F401  (YOLO-World needs it to read your words)
        except ImportError:
            sys.exit("Missing the text model package. Run:\n  pip install git+https://github.com/ultralytics/CLIP.git")
        model = YOLOWorld(args.model)
        model.set_classes(args.words)
    else:
        model = YOLO(args.model)          # an ordinary model: minifigures are found as "person"
        classes = [k for k, v in model.names.items() if v == "person"] or None
        args.words = ["person"]

    print("Using", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "the CPU (slow)")
    image = cv2.imread(str(photo))
    if image is None:
        sys.exit(f"Can't open {photo}. Use a JPG or PNG.")
    found = detect(model, image, args.tiles, args.imgsz, args.conf, classes)

    out = image.copy()
    t = max(2, image.shape[1] // 600)
    for i, (x0, y0, x1, y1, s) in enumerate(found, 1):
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(out, p0, p1, (255, 120, 0), t)
        cv2.putText(out, str(i), (p0[0] + 2, p0[1] + 12 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (0, 220, 255), t)
    path = photo.with_name(photo.stem + "_world.jpg")
    cv2.imwrite(str(path), out)
    if args.crops:
        folder = photo.with_name(photo.stem + "_figures")
        folder.mkdir(exist_ok=True)
        for i, (x0, y0, x1, y1, s) in enumerate(found, 1):
            pad = 0.08 * max(x1 - x0, y1 - y0)
            crop = image[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)]
            cv2.imwrite(str(folder / f"{i:03d}.jpg"), crop)
        print(f"Saved {len(found)} crops to {folder}")
    n = len(found)
    print(f"Found {n} match{'es' if n != 1 else ''} for {', '.join(repr(w) for w in args.words)} "
          f"(confidence {args.conf}, tiles {args.tiles}).")
    print(f"Saved {path}. Open it to see the numbered boxes.")


if __name__ == "__main__":
    main()
