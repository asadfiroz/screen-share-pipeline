import os, glob, random, csv
import numpy as np
import cv2

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    from PIL import Image
    HEIF_OK = True
except Exception:
    HEIF_OK = False

NUM_SAMPLES    = int(os.environ.get("NUM_SAMPLES", 2000))
IMG_SIZE       = 256
OUTPUT_DIR     = "dataset"
CONTENT_DIR    = "content"
BACKGROUND_DIR = "backgrounds"
DEBUG_OVERLAYS = 12
random.seed(42); np.random.seed(42)


def list_images(folder):
    if not os.path.isdir(folder): return []
    files = set()
    for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.heic","*.heif",
              "*.JPG","*.JPEG","*.PNG","*.HEIC","*.HEIF"):
        files.update(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif"):
        if not HEIF_OK: return None
        try:
            img = Image.open(path).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return cv2.imread(path)


CONTENT = list_images(CONTENT_DIR)
BACKGROUNDS = list_images(BACKGROUND_DIR)


def procedural_content(w, h):
    img = np.full((h, w, 3), random.randint(170, 255), np.uint8)
    for _ in range(random.randint(10, 28)):
        p1 = (random.randint(0, w-1), random.randint(0, h-1))
        p2 = (random.randint(0, w-1), random.randint(0, h-1))
        color = tuple(random.randint(0, 255) for _ in range(3))
        if random.random() < 0.5: cv2.rectangle(img, p1, p2, color, -1)
        else: cv2.line(img, p1, p2, color, random.randint(1, 4))
    return img


def procedural_background(size):
    base = np.random.randint(40, 200, (2, 2, 3), np.uint8)
    bg = cv2.resize(base, (size, size), interpolation=cv2.INTER_LINEAR)
    noise = np.random.randint(-15, 15, (size, size, 3), np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def get_content():
    if CONTENT:
        im = load_image(random.choice(CONTENT))
        if im is not None: return im
    w = random.randint(300, 700); h = int(w * random.uniform(0.55, 0.72))
    return procedural_content(w, h)


def get_background():
    if BACKGROUNDS:
        im = load_image(random.choice(BACKGROUNDS))
        if im is not None: return cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    return procedural_background(IMG_SIZE)


def make_keyboard(w, h):
    """A dark keyboard deck with a faint key grid and a trackpad."""
    deck = np.full((h, w, 3), random.randint(22, 55), np.uint8)
    rows, cols = random.randint(4, 6), random.randint(10, 16)
    mx, my = int(w*0.06), int(h*0.12)
    gw = (w - 2*mx) / cols
    gh = (h*0.58 - 2*my) / rows
    kc = random.randint(55, 110)
    for r in range(rows):
        for c in range(cols):
            x, y = int(mx + c*gw), int(my + r*gh)
            cv2.rectangle(deck, (x+1, y+1), (int(x+gw-2), int(y+gh-2)), (kc, kc, kc), -1)
    tw, th = int(w*0.30), int(h*0.26)
    tx, ty = (w - tw)//2, int(h*0.68)
    cv2.rectangle(deck, (tx, ty), (tx+tw, ty+th), (70, 70, 70), -1)
    return deck


def build_panel(content):
    """Assemble a laptop panel (display + bezel + optional keyboard).
    Returns: canvas(BGR), mask(uint8), display_corners(4x2 TL,TR,BR,BL)."""
    dw_in = random.randint(360, 680)
    ar = random.uniform(1.45, 1.80)
    dh_in = int(dw_in / ar)
    content = cv2.resize(content, (dw_in, dh_in))

    b = max(4, int(min(dw_in, dh_in) * random.uniform(0.03, 0.07)))   # bezel
    screen_w, screen_h = dw_in + 2*b, dh_in + 2*b

    has_kb = random.random() < 0.75
    if has_kb:
        kw = int(screen_w * random.uniform(1.00, 1.12))
        kh = int(dh_in * random.uniform(0.45, 0.80))
        hinge = max(2, int(b * 0.8))
    else:
        kw = kh = hinge = 0

    panel_w = max(screen_w, kw)
    panel_h = screen_h + hinge + kh
    canvas = np.zeros((panel_h, panel_w, 3), np.uint8)
    mask = np.zeros((panel_h, panel_w), np.uint8)

    sx = (panel_w - screen_w) // 2
    canvas[0:screen_h, sx:sx+screen_w] = (random.randint(0, 28),) * 3   # bezel
    canvas[b:b+dh_in, sx+b:sx+b+dw_in] = content                         # display
    mask[0:screen_h, sx:sx+screen_w] = 255
    disp = np.array([[sx+b, b], [sx+b+dw_in, b],
                     [sx+b+dw_in, b+dh_in], [sx+b, b+dh_in]], np.float32)

    if has_kb:
        kx, ky = (panel_w - kw)//2, screen_h + hinge
        canvas[screen_h:ky, sx:sx+screen_w] = (random.randint(0, 25),) * 3  # hinge
        mask[screen_h:ky, sx:sx+screen_w] = 255
        canvas[ky:ky+kh, kx:kx+kw] = make_keyboard(kw, kh)
        mask[ky:ky+kh, kx:kx+kw] = 255

    return canvas, mask, disp


def augment(img):
    img = np.clip(img.astype(np.float32) * random.uniform(0.55, 1.25), 0, 255).astype(np.uint8)
    if random.random() < 0.55:
        k = random.choice([3, 5]); img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.6:
        n = np.random.randint(-12, 12, img.shape, np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def make_sample():
    canvas, mask, disp = build_panel(get_content())
    panel_h, panel_w = canvas.shape[:2]

    # place the laptop into the scene at a random size + position + tilt
    max_h = random.uniform(0.35, 0.95) * IMG_SIZE
    scale = max_h / panel_h
    if panel_w * scale > 0.98 * IMG_SIZE:
        scale = 0.98 * IMG_SIZE / panel_w
    dw, dh = panel_w * scale, panel_h * scale
    cx = random.uniform(dw/2 + 2, IMG_SIZE - dw/2 - 2)
    cy = random.uniform(dh/2 + 2, IMG_SIZE - dh/2 - 2)
    x0, x1, y0, y1 = cx - dw/2, cx + dw/2, cy - dh/2, cy + dh/2
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    j = min(dw, dh) * random.uniform(0.04, 0.16)
    dst += np.random.uniform(-j, j, dst.shape).astype(np.float32)
    src = np.array([[0, 0], [panel_w, 0], [panel_w, panel_h], [0, panel_h]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(canvas, M, (IMG_SIZE, IMG_SIZE))
    wmask  = cv2.warpPerspective(mask,   M, (IMG_SIZE, IMG_SIZE))
    comp = get_background().copy()
    comp[wmask > 0] = warped[wmask > 0]
    comp = augment(comp)
    label = cv2.perspectiveTransform(disp.reshape(-1, 1, 2), M).reshape(4, 2)
    return comp, label


def main():
    if BACKGROUNDS or CONTENT:
        heic = [f for f in (BACKGROUNDS + CONTENT) if f.lower().endswith((".heic",".heif"))]
        if heic and not HEIF_OK:
            print("WARNING: HEIC files found but pillow-heif is not installed.")
            print("         Run: pip install pillow-heif   (those files are being skipped)")
    print(f"content images: {len(CONTENT)} | background images: {len(BACKGROUNDS)}")
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "debug"), exist_ok=True)
    rows = []
    for i in range(NUM_SAMPLES):
        img, label = make_sample()
        name = f"img_{i:05d}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", name), img)
        rows.append([name] + [f"{v:.5f}" for v in (label / IMG_SIZE).reshape(-1)])
        if i < DEBUG_OVERLAYS:
            dbg = img.copy(); pts = label.astype(int)
            cv2.polylines(dbg, [pts], True, (0, 255, 0), 2)
            for (x, y) in pts: cv2.circle(dbg, (x, y), 4, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "debug", name), dbg)
    with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","x0","y0","x1","y1","x2","y2","x3","y3"])
        w.writerows(rows)
    print(f"Generated {NUM_SAMPLES} images -> {OUTPUT_DIR}/images")
    print(f"Labels -> {OUTPUT_DIR}/labels.csv | debug overlays -> {OUTPUT_DIR}/debug")


if __name__ == "__main__":
    main()