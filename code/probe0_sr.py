"""
Probe 0 SR engine — minimal Real-ESRGAN (basicsr RRDBNet) loader.

Why hand-rolled instead of `pip install realesrgan`
    basicsr / realesrgan pin old torch + numpy and would fight the emssm env
    (torch 2.11.0+cu128, numpy 2.4.6).  The architecture is ~70 lines, so the
    probe loads the published weights directly and keeps the environment intact.

Weights (fetched from the domestic HF mirror by fetch_weights()):
    ai-forever/Real-ESRGAN  RealESRGAN_x2.pth / x4.pth / x8.pth
    -> basicsr RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2/4/8)

Self-test
    python probe0_sr.py --selftest        # prints the realised scale factor
"""
import os
import sys
import io
import argparse
import urllib.request as U

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MIRROR = "https://hf-mirror.com"
WEIGHTS_SUBDIR = "_weights"

# verified against the published files (see _sr_verify.txt):
#   RealESRGAN_x4plus.pth  3ch, 23 RRDB, conv_up1+conv_up2  -> x4   [canonical]
#   RealESRGAN_x8.pth      3ch, 23 RRDB, conv_up1..3        -> x8
#   RealESRGAN_x2.pth      12ch input -> UNUSABLE for plain RGB, excluded
SR_SOURCES = {
    4: ("schwgHao/RealESRGAN_x4plus", "RealESRGAN_x4plus.pth"),
    8: ("ai-forever/Real-ESRGAN", "RealESRGAN_x8.pth"),
}

# ---------------------------------------------------------------------------
# Second engine: BSRGAN.  Released by Zhang et al. and trained with an L1
# objective plus explicit degradation modelling -- i.e. a FIDELITY-oriented
# engine, not a perceptual GAN.  It is the control that separates two very
# different claims: "the perceptual prior is what hurts detection" versus
# "super-resolution along this path cannot help".
#
# Its state dict uses the BasicSR naming, which maps onto the RRDBNet built
# below through a pure key rename (shapes already match 1:1):
#     RRDB_trunk.{i}.RDB{j}.conv{k}  ->  body.{i}.rdb{j}.conv{k}
#     trunk_conv  -> conv_body      upconv1 -> conv_up1
#     upconv2     -> conv_up2       HRconv  -> conv_hr
# conv_first / conv_last keep their names.
BSRGAN_SOURCE = ("uwg/upscaler", "ESRGAN/BSRGAN.pth")
BSRGAN_RENAME = [("RRDB_trunk.", "body."), (".RDB", ".rdb"),
                 ("trunk_conv.", "conv_body."), ("upconv1.", "conv_up1."),
                 ("upconv2.", "conv_up2."), ("HRconv.", "conv_hr.")]


def _rename_bsrgan(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        for a, b in BSRGAN_RENAME:
            nk = nk.replace(a, b)
        out[nk] = v
    return out


ENGINES = ("realesrgan", "bsrgan")
YOLO_REPO = "Ultralytics/YOLOv8"
YOLO_FILES = ["yolov8n.pt", "yolov8s.pt"]

N_UP = {4: 2, 8: 3}                # 2**n_up == scale


def workdir():
    return os.path.dirname(os.path.abspath(__file__))


def fetch(repo, fname, subdir=WEIGHTS_SUBDIR, quiet=False):
    """Mirror-first download. Returns the local path, or None on failure."""
    dest = os.path.join(workdir(), subdir, fname.replace("/", "__"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = f"{MIRROR}/{repo}/resolve/main/{fname}"
    try:
        with U.urlopen(U.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                       timeout=180) as r, open(dest, "wb") as f:
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                f.write(c)
        if not quiet:
            print(f"[fetch] {fname} <- {MIRROR}  ({os.path.getsize(dest):,} B)")
        return dest
    except Exception as e:
        if not quiet:
            print(f"[fetch] FAILED {url} -> {type(e).__name__}: {e}")
        return None


def fetch_weights(scales=(4, 8), yolo=("yolov8n.pt",)):
    got = {}
    for s in scales:
        if s not in SR_SOURCES:
            print(f"[fetch] scale x{s} has no verified source, skipped")
            continue
        repo, fn = SR_SOURCES[s]
        p = fetch(repo, fn)
        if p:
            got[f"sr_x{s}"] = p
    for y in yolo:
        p = fetch(YOLO_REPO, y)
        if p:
            got[y] = p
    return got


# --------------------------------------------------------------------------
def build_rrdbnet(scale, nf=64, nb=23, gc=32):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf, gc=32):
            super().__init__()
            self.rdb1 = RDB(nf, gc)
            self.rdb2 = RDB(nf, gc)
            self.rdb3 = RDB(nf, gc)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, n_up):
            super().__init__()
            self.n_up = n_up
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            if n_up >= 3:
                self.conv_up3 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(
                F.interpolate(feat, scale_factor=2, mode="nearest")))
            if self.n_up >= 2:
                feat = self.lrelu(self.conv_up2(
                    F.interpolate(feat, scale_factor=2, mode="nearest")))
            if self.n_up >= 3:
                feat = self.lrelu(self.conv_up3(
                    F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet(N_UP[scale])


class RealESRGAN:
    """Loads a published RRDBNet and runs tiled float inference on uint8 images."""

    def __init__(self, scale, device="cuda", weights=None, tile=512, half=True,
                 engine="realesrgan"):
        import torch
        self.scale = scale
        self.tile = tile
        self.engine = engine
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = build_rrdbnet(scale)
        if weights:
            wp = weights
        elif engine == "bsrgan":
            wp = fetch(BSRGAN_SOURCE[0], BSRGAN_SOURCE[1])
            if not wp:
                raise SystemExit("could not obtain the BSRGAN weights")
        else:
            wp = os.path.join(workdir(), WEIGHTS_SUBDIR, SR_SOURCES[scale][1])
        self.weights_path = wp
        sd = torch.load(wp, map_location="cpu", weights_only=True)
        for k in ("params_ema", "params"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        if engine == "bsrgan":
            sd = _rename_bsrgan(sd)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        self.load_report = dict(missing=list(missing), unexpected=list(unexpected))
        if missing or unexpected:
            print(f"[sr] load report {engine} x{scale}: missing={len(missing)} "
                  f"unexpected={len(unexpected)}")
            for k in (missing[:5] + unexpected[:5]):
                print("     ", k)
        self.half = bool(half) and str(self.device).startswith("cuda")
        self.net.eval()
        if self.half:
            self.net.half()
        self.net.to(self.device)

    def _run(self, arr):
        """arr: HxWx3 uint8 -> H*sc x W*sc x3 uint8."""
        import torch
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)[None]
        t = t.to(self.device)
        if self.half:
            t = t.half()
        with torch.no_grad():
            y = self.net(t)
        y = y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return (y * 255.0 + 0.5).astype(np.uint8)

    def upscale(self, img):
        """img: HxWx3 uint8 (or HxW / HxWxC) -> upscaled uint8, same channel count."""
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
            single = True
        else:
            single = img.shape[2] == 1
            if single:
                img = np.repeat(img, 3, 2)
        h, w = img.shape[:2]
        t = self.tile
        out = None
        s = self.scale
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                blk = img[y0:y0 + t, x0:x0 + t]
                try:
                    r = self._run(np.ascontiguousarray(blk))
                except Exception:
                    r = self._run(np.ascontiguousarray(blk[: max(1, t // 2), : max(1, t // 2)]))
                if out is None:
                    out = np.zeros((h * s, w * s, 3), dtype=np.uint8)
                bh, bw = r.shape[:2]
                out[y0 * s:y0 * s + bh, x0 * s:x0 * s + bw] = r[: min(bh, out.shape[0] - y0 * s),
                                                               : min(bw, out.shape[1] - x0 * s)]
        return out[:, :, :1] if single else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scales", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--engine", default="realesrgan", choices=list(ENGINES))
    ap.add_argument("--real-image", default="",
                    help="optional image path; runs an end-to-end x4 upsample")
    args = ap.parse_args()

    if not args.selftest:
        print(__doc__)
        return

    if args.engine == "bsrgan":
        paths = {}
        ok = True
        for s in args.scales:
            if s != 4:
                print(f"[{args.engine}] only x4 is available, skipping x{s}")
                continue
            eng = RealESRGAN(s, tile=128, engine="bsrgan")
            x = (np.random.default_rng(0).integers(0, 255, (32, 48, 3))).astype(np.uint8)
            y = eng.upscale(x)
            realised = y.shape[0] / x.shape[0]
            flag = "OK" if abs(realised - s) < 1e-6 else "MISMATCH"
            if flag != "OK":
                ok = False
            print(f"[{args.engine} x{s}] in {x.shape} -> out {y.shape} "
                  f"realised x{realised:.3f}  {flag}  "
                  f"missing={len(eng.load_report['missing'])} "
                  f"unexpected={len(eng.load_report['unexpected'])}")
        print("SELFTEST", "PASS" if ok else "FAIL")
        return

    paths = fetch_weights(scales=tuple(args.scales), yolo=())
    ok = True
    for s in args.scales:
        p = paths.get(f"sr_x{s}")
        if not p:
            print(f"[x{s}] weights unavailable")
            ok = False
            continue
        eng = RealESRGAN(s, weights=p, tile=128)
        x = (np.random.default_rng(0).integers(0, 255, (32, 48, 3))).astype(np.uint8)
        y = eng.upscale(x)
        realised = y.shape[0] / x.shape[0]
        flag = "OK" if abs(realised - s) < 1e-6 else "MISMATCH"
        if flag != "OK":
            ok = False
        print(f"[x{s}] in {x.shape} -> out {y.shape}  realised x{realised:.3f}  {flag}")

    if args.real_image and os.path.exists(args.real_image):
        import cv2
        img = cv2.imread(args.real_image, cv2.IMREAD_COLOR)
        eng = RealESRGAN(4, tile=256)
        up = eng.upscale(img)
        out_png = os.path.join(workdir(), "_sr_smoke_x4.png")
        cv2.imwrite(out_png, up)
        print(f"[real] {img.shape} -> {up.shape}  wrote {out_png}")
        # a second pass through bicubic for a side-by-side reference
        bic = cv2.resize(img, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(workdir(), "_sr_smoke_bicubic.png"), bic)

    print("SELFTEST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
