"""Kendi el yazisi rakamlarini MNIST formatina cevirip egitilmis MLP ile tahmin eder.

Iki kullanim sekli:
  1) Her rakam ayri dosya:   python predict_custom.py custom_digits/2_a.png custom_digits/4_a.png
     (dosya adinin basi gercek rakam olsun: 2_a.png -> gercek etiket 2)
  2) Tum numara tek gorselde: python predict_custom.py numara.png --segment --truth 24120205036
     (--truth verilirse rakam sayisi bilinir; birbirine degen rakamlar bu sayiya gore ayrilir.
      Gercek numara yoksa --digits 11 gibi yalnizca rakam sayisi verilebilir.)

Onisleme (MNIST'e benzetme):
  EXIF dondurmesini uygula (telefon fotograflari) -> gri tonlama
  -> arka plan (golge, kagit dokusu) cikarma -> kalem/cizgi ayirma (cizgili kagit temizligi)
  -> metin satirini bul -> rakamlari bosluklardan ayir (degenleri rakam sayisina gore bol)
  -> rakamin sinirlarina kirp -> ince cizgiyi kalinlastir
  -> en-boy orani korunarak 20x20'ye sigdir -> kutle merkezine gore 28x28 tuvalin ortasina koy
  -> [0,1] olcekle -> MNIST ortalama/std ile normalizasyon
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def otsu_threshold(u8):
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = (np.arange(256) * hist).sum()
    w_b = sum_b = 0.0
    best, best_t = -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / (total - w_b)
        var = w_b * (total - w_b) * (m_b - m_f) ** 2
        if var > best:
            best, best_t = var, t
    return best_t


def load_gray(path):
    """Fotografi acar, EXIF yonunu duzeltir (telefon dikey cekimleri), gri tonlamaya cevirir."""
    img = ImageOps.exif_transpose(Image.open(path))
    return np.asarray(ImageOps.grayscale(img), dtype=np.uint8)


def extract_ink(gray):
    """gray: 2B uint8 fotograf -> 2B uint8 'murekkep' haritasi (kalem parlak, zemin 0).

    1) Buyuk yaricapli bulaniklastirma ile kagidin aydinlik/golge haritasi cikarilir.
    2) Kalem (koyu, guclu) ile kagit cizgileri/leke (soluk) ikinci bir Otsu esigiyle ayrilir;
       soluk pikseller yalnizca guclu kalem piksellerinin yakininda tutulur.
    3) En yogun kalem satiri (metin bandi) bulunur, bandin disi silinir.
    """
    img = Image.fromarray(gray)
    if max(img.size) > 1000:                         # cok buyuk fotografi kucult
        s = 1000 / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                         Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    if a.mean() < 127:                               # koyu zemin, acik kalem -> kalem koyu olsun
        a = 255 - a
    a_img = Image.fromarray(a.astype(np.uint8))
    radius = max(15, min(a.shape) // 8)
    bg = np.asarray(a_img.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(radius)),
                    dtype=np.float32)
    # bagil kontrast: golgedeki (karanlik) bolgede de kalem ayni siddette gorunsun
    diff = np.clip(bg - a, 0, None) / np.maximum(bg, 20.0)
    if diff.max() <= 0:
        return np.zeros(a.shape, np.uint8)
    diff = (diff / diff.max() * 255).astype(np.uint8)
    t1 = max(otsu_threshold(diff), 50)
    ink = np.where(diff > t1, diff, 0).astype(np.uint8)

    fg = ink[ink > 0]
    if fg.size < 50:
        return ink
    # ikinci seviye: guclu kalem cekirdegi / soluk cizgi ayrimi
    t2 = otsu_threshold(fg)
    strong = ink > t2
    if strong.sum() < 30:
        return ink
    near = np.asarray(Image.fromarray((strong * 255).astype(np.uint8))
                      .filter(ImageFilter.MaxFilter(9))) > 0
    ink = np.where(near, ink, 0).astype(np.uint8)

    # metin bandi: en cok guclu kalem iceren satir grubu
    H = ink.shape[0]
    prof = strong.sum(axis=1)
    on = np.nonzero(prof >= 0.1 * prof.max())[0]
    gap = max(3, int(0.02 * H))
    groups, start, prev = [], on[0], on[0]
    for y in on[1:]:
        if y - prev > gap:
            groups.append((start, prev))
            start = y
        prev = y
    groups.append((start, prev))
    y0, y1 = max(groups, key=lambda g: prof[g[0]:g[1] + 1].sum())
    pad = max(5, int(0.01 * H))
    out = np.zeros_like(ink)
    out[max(0, y0 - pad):y1 + pad + 1] = ink[max(0, y0 - pad):y1 + pad + 1]
    return out


def _runs(on):
    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(on)])
    return runs


def segment_digits(ink, n_expected=None):
    """Kolon projeksiyonuyla yan yana yazilmis rakamlari ayirir.

    n_expected verilirse: parca sayisi fazlaysa en yakin parcalar birlestirilir,
    azsa (rakamlar degiyorsa) en genis parcalar kolon yogunlugunun en dusuk oldugu
    yerden bolunur. Doner: (rakam goruntuleri, [(bas, son), ...] kolon sinirlari).
    """
    mask = ink > 0
    prof = mask.sum(axis=0)
    rows = np.nonzero(mask.any(axis=1))[0]
    band_h = 0 if len(rows) == 0 else rows.max() - rows.min() + 1
    runs = _runs(prof >= max(1, int(0.02 * band_h)))      # tek tuk kalan gurultu kolonlarini say
    runs = [r for r in runs if mask[:, r[0]:r[1]].sum() >= 15]      # nokta/leke

    def row_extent(s, e):
        rows = np.nonzero(mask[:, s:e].any(axis=1))[0]
        return 0 if len(rows) == 0 else rows.max() - rows.min() + 1
    if runs:
        ext = [row_extent(s, e) for s, e in runs]
        runs = [r for r, x in zip(runs, ext) if x >= 0.3 * max(ext)]

    if n_expected and runs:
        while len(runs) > n_expected:                                # kopuk rakamlari birlestir
            gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
            i = int(np.argmin(gaps))
            runs[i] = [runs[i][0], runs[i + 1][1]]
            del runs[i + 1]
        if len(runs) < n_expected:                                   # degen rakamlari bol
            parts = [1] * len(runs)
            for _ in range(n_expected - len(runs)):
                widths = [(r[1] - r[0]) / p for r, p in zip(runs, parts)]
                parts[int(np.argmax(widths))] += 1
            new = []
            for (s, e), p in zip(runs, parts):
                if p == 1:
                    new.append([s, e])
                    continue
                w = (e - s) / p
                cuts = [s]
                for k in range(1, p):
                    c = s + k * w
                    lo = max(int(c - 0.25 * w), cuts[-1] + 1)
                    hi = min(int(c + 0.25 * w) + 1, e - 1)
                    cuts.append(lo + int(np.argmin(prof[lo:hi])) if hi > lo else int(c))
                cuts.append(e)
                new += [[cuts[i], cuts[i + 1]] for i in range(p)]
            runs = new
    return [ink[:, s:e] for s, e in runs], [tuple(r) for r in runs]


def _render(ink):
    """Kirpilmis rakami (uint8) 20x20'ye sigdirip kutle merkezine gore 28x28 tuvale koyar."""
    ink = np.pad(ink, 2)                           # kenardaki kalem kesilmesin
    h, w = ink.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = np.asarray(Image.fromarray(ink).resize((nw, nh), Image.LANCZOS), dtype=np.float32)
    canvas = np.zeros((28, 28), np.float32)
    # kutle merkezini tuvalin ortasina getir (MNIST boyle hazirlanmistir)
    total = max(img.sum(), 1e-6)
    cy = (img.sum(1) * np.arange(nh)).sum() / total
    cx = (img.sum(0) * np.arange(nw)).sum() / total
    oy = int(np.clip(round(14 - cy), 0, 28 - nh))
    ox = int(np.clip(round(14 - cx), 0, 28 - nw))
    canvas[oy:oy + nh, ox:ox + nw] = img
    return canvas / max(1.0, canvas.max())


def stroke_halfwidth(crop):
    """Cizgi yari kalinligi (piksel): kalem maskesi bitene kadar gereken 3x3 asindirma sayisi."""
    cur = Image.fromarray(((crop > 0.5 * crop.max()) * 255).astype(np.uint8))
    r = 0
    while r < crop.shape[0] and np.asarray(cur).any():
        cur = cur.filter(ImageFilter.MinFilter(3))
        r += 1
    return r


def to_mnist(ink, rel_halfwidth=0.08):
    """ink: 2B uint8 (kalem parlak, zemin 0, tek rakam) -> 28x28 float32 [0,1].

    MNIST'te cizgi kalinligi rakam boyunun ~%15'i kadardir. Foto ince kalemle cekildiyse
    cizgi bu orana gelene kadar kalinlastirilir; zaten yeterince kalinsa dokunulmaz.
    Kalinlik rakamin yogunluguna degil cizgi genisligine bakilarak ayarlanir (dar rakamlar
    ornegin '1' sisip leke olmasin).
    """
    ink = ink.astype(np.uint8).copy()
    ink[ink < 0.15 * max(1, ink.max())] = 0
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    target = max(1, round(rel_halfwidth * crop.shape[0]))
    pil = Image.fromarray(crop)
    for _ in range(max(0, target - stroke_halfwidth(crop))):
        pil = pil.filter(ImageFilter.MaxFilter(3))
    return _render(np.asarray(pil))


def save_segment_debug(ink, bounds, path):
    rgb = np.stack([ink] * 3, axis=-1).copy()
    for s, e in bounds:
        rgb[:, max(0, s - 1):s + 1] = (255, 0, 0)
        rgb[:, max(0, e - 1):e] = (0, 120, 255)
    ys = np.nonzero((ink > 0).any(axis=1))[0]
    if len(ys):
        rgb = rgb[max(0, ys.min() - 10):ys.max() + 10]
    Path(path).parent.mkdir(exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from mlp_mnist import MLP, MNIST_MEAN, MNIST_STD

    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default="results/deney1_temel_mlp.pt")
    p.add_argument("--segment", action="store_true")
    p.add_argument("--truth", default="", help="--segment icin gercek numara")
    p.add_argument("--digits", type=int, default=0, help="--segment icin rakam sayisi")
    p.add_argument("--out", default="results/ozel_tahminler.png")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    model = MLP(ckpt["hidden"], ckpt.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    samples = []  # (28x28 goruntu, gercek etiket ya da None)
    for path in args.images:
        ink = extract_ink(load_gray(path))
        if args.segment:
            n = args.digits or len(args.truth) or None
            parts, bounds = segment_digits(ink, n)
            save_segment_debug(ink, bounds, "results/segmentler.png")
            print(f"{path}: {len(parts)} rakam parcasi bulundu"
                  + (f" (beklenen {n})" if n else ""))
            if n and len(parts) != n:
                print("UYARI: parca sayisi beklenenle ayni degil. results/segmentler.png dosyasina bakin.")
            for i, d in enumerate(parts):
                t = args.truth[i] if i < len(args.truth) else None
                samples.append((to_mnist(d), t))
        else:
            first = Path(path).stem[0]
            samples.append((to_mnist(ink), first if first.isdigit() else None))

    if not samples:
        raise SystemExit("Hic rakam bulunamadi. Fotografi kontrol edin.")

    x = torch.tensor(np.stack([s[0] for s in samples]))[:, None]
    x = (x - MNIST_MEAN) / MNIST_STD
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)
    pred = prob.argmax(1)

    preds, correct, known = [], 0, 0
    fig, axes = plt.subplots(1, len(samples), figsize=(1.6 * len(samples), 2.2), squeeze=False)
    for ax, (img, truth), pr, pb in zip(axes[0], samples, pred, prob.max(1).values):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{pr.item()} ({pb:.2f})", fontsize=9)
        ax.axis("off")
        preds.append(str(pr.item()))
        if truth is not None:
            known += 1
            correct += int(str(pr.item()) == truth)
    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150)

    print("Tahmin:", "".join(preds))
    if known:
        print(f"Dogru: {correct}/{known}  (accuracy {correct / known:.2%})")
    print("Gorsel:", args.out)


if __name__ == "__main__":
    main()
