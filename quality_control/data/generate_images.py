"""
OC Doors (formerly Masonite International) — Door Skin Defect Image Generator
Defect types: good, crack, blister, crooked_corner, thin_paint,
              thick_paint, scratch, delamination
Products: Interior, Exterior, Fiberglass door skins
"""
import cv2
import numpy as np
import os
import json
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

CLASSES = [
    "good",
    "crack",
    "blister",
    "crooked_corner",
    "thin_paint",
    "thick_paint",
    "scratch",
    "delamination",
]

IMG_SIZE = (224, 224)

SAMPLES = {
    "good":           300,
    "crack":          140,
    "blister":        130,
    "crooked_corner": 110,
    "thin_paint":     130,
    "thick_paint":    120,
    "scratch":        130,
    "delamination":   110,
}

SEVERITY = {
    "good":           0,
    "thin_paint":     1,
    "thick_paint":    1,
    "scratch":        2,
    "blister":        2,
    "crooked_corner": 3,
    "crack":          3,
    "delamination":   3,
}

DEFECT_INFO = {
    "good":           "No defects. Door skin meets all OC Doors quality standards.",
    "crack":          "Structural crack in door skin surface. Critical — door fails structural test.",
    "blister":        "Sub-surface air bubble causing paint/surface to lift. Cosmetic reject.",
    "crooked_corner": "Corner geometry out of tolerance. Door will not fit frame correctly.",
    "thin_paint":     "Insufficient paint thickness. Fails warranty and weather resistance spec.",
    "thick_paint":    "Excessive paint buildup. Causes door binding in frame and appearance defect.",
    "scratch":        "Surface scratch from handling or tooling contact. Cosmetic defect.",
    "delamination":   "Skin separating from substrate. Structural failure — immediate reject.",
}

DEFECT_COST = {
    "good":           0,
    "thin_paint":     35,
    "thick_paint":    35,
    "scratch":        55,
    "blister":        85,
    "crooked_corner": 120,
    "crack":          180,
    "delamination":   200,
}


# ── Base textures ──────────────────────────────────────────────────────────────

def door_skin_texture(h, w, product="interior"):
    """Simulate door skin surface texture — painted MDF/fiberglass/steel."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    base_colors = {
        "interior":   (235, 232, 226),   # off-white primed MDF
        "exterior":   (210, 205, 195),   # primed fiberglass / steel
        "fiberglass": (198, 193, 182),   # fiberglass gel coat
    }
    base = base_colors.get(product, (230, 226, 218))
    img[:] = base

    # Subtle wood-grain-like texture for interior
    if product == "interior":
        for _ in range(random.randint(8, 18)):
            y1 = random.randint(0, h - 1)
            x1, x2 = 0, w
            shade = random.randint(-8, 8)
            col = tuple(max(0, min(255, c + shade)) for c in base)
            cv2.line(img, (x1, y1), (x2, y1 + random.randint(-3, 3)),
                     col, random.randint(1, 3))

    # Panel embossing lines for fiberglass
    if product == "fiberglass":
        for margin in [20, h - 20, w // 3, 2 * w // 3]:
            cv2.line(img, (0, margin), (w, margin), tuple(c - 15 for c in base), 1)

    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def add_lighting(img):
    f = random.uniform(0.86, 1.14)
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


# ── Defect generators ──────────────────────────────────────────────────────────

def make_good(h, w):
    product = random.choice(["interior", "exterior", "fiberglass"])
    return add_lighting(door_skin_texture(h, w, product))


def make_crack(h, w):
    """Hairline or structural crack in door skin."""
    img = door_skin_texture(h, w)
    num_cracks = random.randint(1, 3)
    for _ in range(num_cracks):
        pts = [(random.randint(10, w - 10), random.randint(10, h - 10))]
        angle = random.uniform(0, 360)
        length = random.randint(40, 120)
        for _ in range(random.randint(4, 10)):
            step = random.randint(8, 20)
            angle += random.uniform(-25, 25)
            nx = np.clip(int(pts[-1][0] + step * np.cos(np.radians(angle))), 0, w - 1)
            ny = np.clip(int(pts[-1][1] + step * np.sin(np.radians(angle))), 0, h - 1)
            pts.append((nx, ny))
            if abs(nx - pts[0][0]) + abs(ny - pts[0][1]) > length:
                break
        for i in range(len(pts) - 1):
            # Dark crack line
            cv2.line(img, pts[i], pts[i + 1], (60, 58, 55), random.randint(1, 3))
            # Bright stress highlight alongside
            offset = (pts[i][0] + 2, pts[i][1])
            cv2.line(img, offset, (pts[i + 1][0] + 2, pts[i + 1][1]),
                     (255, 252, 245), 1)
    return add_lighting(img)


def make_blister(h, w):
    """Paint blisters — raised bubbles under paint surface."""
    img = door_skin_texture(h, w)
    num = random.randint(1, 6)
    for _ in range(num):
        cx = random.randint(15, w - 15)
        cy = random.randint(15, h - 15)
        r  = random.randint(6, 22)
        # Outer ring (shadow)
        cv2.circle(img, (cx, cy), r + 3, (170, 166, 158), -1)
        # Blister dome (lighter centre)
        cv2.circle(img, (cx, cy), r, (248, 245, 238), -1)
        # Highlight spot
        cv2.circle(img, (cx - r // 3, cy - r // 3), r // 3,
                   (255, 254, 252), -1)
        # Edge shadow
        cv2.circle(img, (cx, cy), r, (140, 136, 128), 2)
    return add_lighting(img)


def make_crooked_corner(h, w):
    """Corner geometry defect — out-of-square corner."""
    img = door_skin_texture(h, w)
    corner = random.choice(["TL", "TR", "BL", "BR"])
    deviation = random.randint(12, 35)

    if corner == "TL":
        pts = np.array([[0, 0], [deviation + random.randint(0, 15), 0],
                        [0, deviation + random.randint(0, 15)]], dtype=np.int32)
    elif corner == "TR":
        pts = np.array([[w, 0], [w - deviation - random.randint(0, 15), 0],
                        [w, deviation + random.randint(0, 15)]], dtype=np.int32)
    elif corner == "BL":
        pts = np.array([[0, h], [deviation + random.randint(0, 15), h],
                        [0, h - deviation - random.randint(0, 15)]], dtype=np.int32)
    else:
        pts = np.array([[w, h], [w - deviation - random.randint(0, 15), h],
                        [w, h - deviation - random.randint(0, 15)]], dtype=np.int32)

    # Misaligned area
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (185, 180, 170))
    img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    # Edge line showing misalignment
    cv2.polylines(img, [pts], True, (100, 95, 88), 2)
    # Measurement lines
    cv2.line(img, (0, deviation), (deviation * 2, deviation),
             (180, 50, 50), 1)
    return add_lighting(img)


def make_thin_paint(h, w):
    """Thin paint spots — substrate visible through insufficient coverage."""
    img = door_skin_texture(h, w)
    num = random.randint(1, 4)
    for _ in range(num):
        cx = random.randint(20, w - 20)
        cy = random.randint(20, h - 20)
        rx = random.randint(15, 55)
        ry = random.randint(10, 40)
        # Thin area — substrate shows through (darker, different texture)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry),
                    random.uniform(0, 180), 0, 360, 255, -1)
        mb = cv2.GaussianBlur(mask, (31, 31), 12).astype(np.float32) / 255.0
        # Thin paint appears slightly darker and different shade
        substrate = (195, 188, 175)
        for c in range(3):
            img[:, :, c] = np.clip(
                img[:, :, c].astype(np.float32) * (1 - mb * 0.45)
                + substrate[c] * mb * 0.45, 0, 255).astype(np.uint8)
    return add_lighting(img)


def make_thick_paint(h, w):
    """Thick paint spots — orange peel texture, runs, excess buildup."""
    img = door_skin_texture(h, w)
    num = random.randint(1, 3)
    for _ in range(num):
        cx = random.randint(20, w - 20)
        cy = random.randint(20, h - 20)
        rx = random.randint(12, 45)
        ry = random.randint(10, 35)
        # Thick area — lighter and slightly textured
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry),
                    random.uniform(0, 180), 0, 360, 255, -1)
        mb = cv2.GaussianBlur(mask, (25, 25), 10).astype(np.float32) / 255.0
        thick_col = (252, 250, 244)
        for c in range(3):
            img[:, :, c] = np.clip(
                img[:, :, c].astype(np.float32) * (1 - mb * 0.40)
                + thick_col[c] * mb * 0.40, 0, 255).astype(np.uint8)
        # Orange-peel micro texture
        for _ in range(random.randint(8, 20)):
            dx = random.randint(-rx, rx)
            dy = random.randint(-ry, ry)
            px, py = cx + dx, cy + dy
            if dx * dx / (rx * rx) + dy * dy / (ry * ry) < 1 and 0 <= px < w and 0 <= py < h:
                r = random.randint(2, 6)
                b = random.uniform(0.88, 0.97)
                base_col = img[py, px].tolist()
                cv2.circle(img, (px, py), r,
                           tuple(int(c * b) for c in base_col), -1)
    return add_lighting(img)


def make_scratch(h, w):
    """Surface scratch from handling, conveyor, or tooling contact."""
    img = door_skin_texture(h, w)
    num = random.randint(1, 4)
    for _ in range(num):
        x1 = random.randint(5, w - 5)
        y1 = random.randint(5, h - 5)
        length = random.randint(30, 100)
        angle = random.uniform(0, 360)
        x2 = np.clip(int(x1 + length * np.cos(np.radians(angle))), 0, w - 1)
        y2 = np.clip(int(y1 + length * np.sin(np.radians(angle))), 0, h - 1)
        width = random.randint(1, 3)
        # Dark scratch line
        cv2.line(img, (x1, y1), (x2, y2), (80, 78, 74), width)
        # Bright highlight alongside (tool marks leave burnished edge)
        offset = 2
        cv2.line(img, (x1 + offset, y1), (x2 + offset, y2),
                 (252, 250, 246), 1)
    return add_lighting(img)


def make_delamination(h, w):
    """Skin delamination — door skin separating from substrate."""
    img = door_skin_texture(h, w)
    # Large irregular region showing separation
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = random.randint(25, 70)
    ry = random.randint(20, 55)

    # Delaminated area — lifted skin appearance (shadows + highlights)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry),
                random.uniform(0, 90), 0, 360, 255, -1)

    # Irregular edges
    pts_count = random.randint(6, 10)
    pts = []
    for i in range(pts_count):
        a = 2 * np.pi * i / pts_count
        r_var = random.uniform(0.7, 1.3)
        pts.append([int(cx + rx * r_var * np.cos(a)),
                    int(cy + ry * r_var * np.sin(a))])
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts_arr], 255)

    mb = cv2.GaussianBlur(mask, (21, 21), 8).astype(np.float32) / 255.0

    # Shadow around delaminated edge
    delam_col = (155, 150, 140)
    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c].astype(np.float32) * (1 - mb * 0.5)
            + delam_col[c] * mb * 0.5, 0, 255).astype(np.uint8)

    # Lifted edge highlight
    cv2.polylines(img, [pts_arr], True, (90, 85, 78), 3)
    cv2.polylines(img, [pts_arr + 2], True, (248, 245, 238), 1)
    return add_lighting(img)


GENERATORS = {
    "good":           make_good,
    "crack":          make_crack,
    "blister":        make_blister,
    "crooked_corner": make_crooked_corner,
    "thin_paint":     make_thin_paint,
    "thick_paint":    make_thick_paint,
    "scratch":        make_scratch,
    "delamination":   make_delamination,
}


def generate_dataset(output_dir="data/images", split_ratio=0.80, force=False):
    manifest_path = f"{output_dir}/manifest.json"
    if os.path.exists(manifest_path) and not force:
        print(f"Dataset already exists — loading manifest")
        with open(manifest_path) as f:
            return json.load(f)

    h, w = IMG_SIZE
    items = []
    for split in ["train", "val"]:
        for cls in CLASSES:
            Path(f"{output_dir}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

    print(f"Generating {sum(SAMPLES.values())} door skin defect images...")
    for cls in CLASSES:
        n       = SAMPLES[cls]
        n_train = int(n * split_ratio)
        gen     = GENERATORS[cls]
        for i in range(n):
            img   = gen(h, w)
            split = "train" if i < n_train else "val"
            path  = f"{output_dir}/{split}/{cls}/{cls}_{i:04d}.jpg"
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            items.append({"path": path, "label": cls,
                          "split": split, "label_id": CLASSES.index(cls)})
        print(f"  ✅ {cls:<18} {n} images  [{['None','Minor','Moderate','Major'][SEVERITY[cls]]}]")

    manifest = {
        "classes": CLASSES, "samples": SAMPLES,
        "img_size": list(IMG_SIZE), "total": len(items),
        "manifest": items,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ {len(items)} door skin images generated")
    return manifest


if __name__ == "__main__":
    generate_dataset()
