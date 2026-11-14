import csv
import random
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from generate_marksheets import (draw_marksheet, random_values, field_value,
                                 new_field_value, EDIT, DISPLAY, font, INK)

ROOT = HERE.parents[1]
OUT = ROOT / "data" / "raw" / "e3"       # separate from the original dataset
N_TRAIN, N_TEST = 60, 40


# ---------- scan / photo simulation (same recipe for genuine and forged) ----------
def scan_params(rng):
    return {
        "angle": rng.uniform(-0.3, 0.3),      # degrees of rotation
        "blur": rng.uniform(0.4, 1.2),        # Gaussian blur sigma (pixels)
        "noise": rng.uniform(2.0, 8.0),       # Gaussian noise sigma (grey levels)
        "gain": rng.uniform(0.92, 1.08),      # contrast factor
        "bias": rng.uniform(-10, 10),         # brightness shift
        "jpeg": int(rng.integers(60, 91)),    # final JPEG quality
    }


def scan(img, p, rng):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), p["angle"], 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    out = out.astype(np.float32) * p["gain"] + p["bias"]
    out += rng.normal(0, p["noise"], out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_jpg(path, rgb, quality):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


# ---------- helpers for the edits ----------
def box_of(field):
    x, y, size, bold, width = EDIT[field]
    return (x - 10, y - 28, x + width, y + 28)          # x0, y0, x1, y1


def local_background(img, box):
    x0, y0, x1, y1 = box
    px = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    bright = px.mean(axis=1)
    return px[bright >= np.percentile(bright, 70)].mean(axis=0)     # the paper colour here


def render_text(h, w, bg, text, size, bold, scale=1):
    im = Image.new("RGB", (w * scale, h * scale), tuple(int(c) for c in bg))
    ImageDraw.Draw(im).text((10 * scale, h * scale // 2), text, font=font(size * scale, bold),
                            fill=INK, anchor="lm")
    return np.array(im)


# ---------- the three forgery styles ----------
def edit_retype(img, field, text, rng, page_noise):
    """Style 1: cover with local background, type new text, slight blur, less noise than the page."""
    out = img.copy()
    box = box_of(field)
    x0, y0, x1, y1 = box
    patch = render_text(y1 - y0, x1 - x0, local_background(out, box), text, EDIT[field][2], EDIT[field][3])
    patch = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), rng.uniform(0.3, 1.0))
    patch += rng.normal(0, page_noise * rng.uniform(0.2, 0.5), patch.shape)
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def edit_paste(img, field, donor, rng):
    """Style 2: copy the same field from another document's scan, 2 px misalignment, soft edges."""
    out = img.copy()
    x0, y0, x1, y1 = box_of(field)
    dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
    src = donor[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float32)
    h, w = src.shape[:2]
    ramp_y = np.minimum(np.arange(h), np.arange(h)[::-1]) / 4.0
    ramp_x = np.minimum(np.arange(w), np.arange(w)[::-1]) / 4.0
    alpha = np.clip(np.minimum(ramp_y[:, None], ramp_x[None, :]), 0, 1)[..., None]
    out[y0:y1, x0:x1] = (alpha * src + (1 - alpha) * out[y0:y1, x0:x1].astype(np.float32)).astype(np.uint8)
    return out


def edit_resample(img, field, text, rng):
    """Style 3 (unseen in training): text drawn at 2x then shrunk, patch alone JPEG-compressed."""
    out = img.copy()
    box = box_of(field)
    x0, y0, x1, y1 = box
    big = render_text(y1 - y0, x1 - x0, local_background(out, box), text, EDIT[field][2], EDIT[field][3], scale=2)
    patch = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 56))])
    out[y0:y1, x0:x1] = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


# ---------- build one split ----------
def make_split(split, n, styles, seed_base):
    for kind in ("genuine", "manipulated"):
        (OUT / split / kind).mkdir(parents=True, exist_ok=True)

    originals = []
    for i in range(1, n + 1):
        random.seed(seed_base + i)
        values = random_values()
        rng = np.random.default_rng(seed_base + i)
        p = scan_params(rng)
        page = np.array(draw_marksheet(values))
        originals.append({"values": values, "p": p, "scanned": scan(page, p, rng)})

    random.seed(seed_base)
    rng = np.random.default_rng(seed_base)
    rows = []
    for i, o in enumerate(originals, start=1):
        group = f"{split[:2].upper()}{i:03d}"
        genuine_name = f"{split}_{i:03d}.jpg"
        save_jpg(OUT / split / "genuine" / genuine_name, o["scanned"], o["p"]["jpeg"])
        rows.append([genuine_name, "marksheet", 0, group, split, "none"])

        style = styles[i % len(styles)]
        field = str(rng.choice(list(EDIT.keys())))
        if style == "paste":
            forged = None
            for step in range(1, n):
                donor = originals[(i - 1 + step) % n]
                if field_value(donor["values"], field) != field_value(o["values"], field):
                    forged = edit_paste(o["scanned"], field, donor["scanned"], rng)
                    break
        else:
            text = DISPLAY[field].format(new_field_value(o["values"], field))
            if style == "retype":
                forged = edit_retype(o["scanned"], field, text, rng, o["p"]["noise"])
            else:
                forged = edit_resample(o["scanned"], field, text, rng)
        fake_name = f"{split}_{i:03d}_edit1.jpg"
        save_jpg(OUT / split / "manipulated" / fake_name, forged, o["p"]["jpeg"])   # same JPEG quality as its twin
        rows.append([fake_name, "marksheet", 1, group, split, style])

    with open(OUT / f"labels_{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "doc_type", "label", "group_id", "split", "edit_style"])
        writer.writerows(rows)
    styles_used = sorted({r[5] for r in rows if r[2] == 1})
    print(f"{split}: {n} genuine + {n} manipulated, forgery styles: {styles_used}")


if __name__ == "__main__":
    make_split("train", N_TRAIN, ["retype", "paste"], 1000)
    make_split("test", N_TEST, ["resample"], 5000)
    print("Saved in", OUT)