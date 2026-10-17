"""
A2 退化算子库 —— 预注册设计见 E:\\新方向研究\\a2_degradation_probe_design.md

命题：退化管线的改进，是否真的改善真实场景的下游性能，还是只改善合成分数？

六个臂（唯一变量 = 退化算子；其余全固定）
  D0 interarea  : INTER_AREA x4           —— P0-B 的现状（基线）
  D1 bicubic    : 双三次 x4               —— 最常见的合成退化
  D2 highorder  : 模糊 -> 缩放 -> 噪声 -> JPEG（Real-ESRGAN 做法）
  D3 shuffled   : 上述四步随机顺序与随机参数（BSRGAN 做法）
  D4 physical   : 由 MTF 反解 Gaussian PSF -> 面积降采样 -> 传感器噪声（综述 7.1 推荐）
  D5 aniso      : 椭圆/方向性 PSF（模拟平台运动与光学非对称）

设计纪律
  * D4/D5 的 PSF 参数**由 MTF 反解，不做网格搜索**（否则引入专家偏差）
  * 所有算子接受 uint8 HxWx3 或 HxW，返回同类型、同 scale 缩小的图
  * 自测：`python deg_ops.py` 会打印 MTF 反解的 sigma 与实测 MTF
"""
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ARMS = ("D0_interarea", "D1_bicubic", "D2_highorder", "D3_shuffled",
        "D4_physical", "D5_aniso")

# ── 传感器设定（AI-TOD 航拍，0.5 m GSD）────────────────────────────
GSD_HR = 0.5          # m / px（AI-TOD 原图）
MTF_AT_NYQ = 0.25     # 光学系统在原始 Nyquist 处的 MTF（典型 0.2–0.3，取中值）
SIGMA_READ = 2.0 / 255.0   # 读出噪声 std（8bit 归一化）
SHOT_K = 0.008             # 散粒噪声系数（信号相关）


def mtf_to_sigma(mtf_at_nyq=MTF_AT_NYQ):
    """Gaussian PSF 的 MTF 为 exp(-2*pi^2*sigma^2*f^2)。
    给定 MTF(f_nyq)=M 反解 sigma（单位：原始像素）。
    f_nyq = 0.5 cyc/px。"""
    f = 0.5
    m = float(np.clip(mtf_at_nyq, 1e-3, 0.999))
    return float(np.sqrt(-np.log(m) / (2.0 * np.pi ** 2 * f ** 2)))


def _as_float(img):
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def _to_uint8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ── 基础算子 ────────────────────────────────────────────────────
def deg_interarea(img, scale=4):
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)


def deg_bicubic(img, scale=4):
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def add_sensor_noise(x, rng):
    """信号相关散粒噪声 + 读出噪声。x 为 [0,1] float。"""
    shot = rng.normal(0.0, 1.0, x.shape).astype(np.float32) * np.sqrt(SHOT_K * np.maximum(x, 1e-6))
    read = rng.normal(0.0, SIGMA_READ, x.shape).astype(np.float32)
    return np.clip(x + shot + read, 0.0, 1.0)


def jpeg_roundtrip(img, q):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _gauss_blur(img, sigma):
    import cv2
    if sigma <= 1e-3:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def _aniso_resample_kernel(s0, k, theta, taps=3):
    """椭圆高斯**重采样核**。

    v2 关键设计：几何平均 sigma 取 4×4 盒滤波的等效值（4/sqrt(12) ≈ 1.1547），
    因此 k=1 时该核 ≈ INTER_AREA；k>1 时**唯一差别就是各向异性**。
    核尺寸取 4*taps+1，保证覆盖 4× 下采样所需的抗混叠带宽。
    """
    size = 4 * taps + 1
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * xx + st * yy
    yr = -st * xx + ct * yy
    sx, sy = s0 * np.sqrt(k), s0 / np.sqrt(k)
    K = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return (K / K.sum()).astype(np.float32)


def _aniso_resample(img, scale, k, theta, s0, taps=3):
    """椭圆核滤波后按 scale 抽样。核已含抗混叠，故直接取点即可。"""
    import cv2
    K = _aniso_resample_kernel(s0, k, theta, taps)
    x = cv2.filter2D(img, -1, K, borderType=cv2.BORDER_REFLECT)
    x = x[::scale, ::scale]
    return x


# ── 六个臂 ──────────────────────────────────────────────────────
def deg_highorder(img, scale=4, rng=None, q_range=(30, 95)):
    """Real-ESRGAN 式高阶退化：blur -> resize -> noise -> JPEG，固定顺序。"""
    rng = rng or np.random.default_rng(0)
    x = _as_float(img)
    x = _gauss_blur(x, float(rng.uniform(0.2, 1.2)))
    import cv2
    h, w = x.shape[:2]
    x = cv2.resize(x, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    x = add_sensor_noise(x, rng)
    x = _to_uint8(x)
    x = jpeg_roundtrip(x, rng.integers(q_range[0], q_range[1] + 1))
    return x


def deg_shuffled(img, scale=4, rng=None):
    """BSRGAN 式乱序退化：四步顺序与参数均随机。"""
    rng = rng or np.random.default_rng(0)
    x = _as_float(img)
    import cv2
    h, w = x.shape[:2]
    steps = ["blur", "resize", "noise", "jpeg"]
    rng.shuffle(steps)
    for s in steps:
        if s == "blur":
            x = _gauss_blur(x, float(rng.uniform(0.0, 1.6)))
        elif s == "resize":
            cur_h, cur_w = x.shape[:2]
            if cur_w // scale >= 2:
                x = cv2.resize(x, (cur_w // scale, cur_h // scale),
                               interpolation=int(rng.choice(
                                   [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])))
            if rng.random() < 0.35:   # 随机升回，制造二次退化
                x = cv2.resize(x, (x.shape[1] * scale, x.shape[0] * scale),
                               interpolation=cv2.INTER_LINEAR)
        elif s == "noise":
            x = add_sensor_noise(np.clip(x, 0, 1), rng)
        elif s == "jpeg":
            x = _as_float(jpeg_roundtrip(_to_uint8(np.clip(x, 0, 1)),
                                         int(rng.integers(28, 96))))
    while x.shape[1] > w // scale:      # 保证最终倍率一致
        x = cv2.resize(x, (x.shape[1] // scale, x.shape[0] // scale),
                       interpolation=cv2.INTER_AREA)
    return _to_uint8(np.clip(x, 0, 1))


def deg_physical(img, scale=4, rng=None, mtf=MTF_AT_NYQ):
    """物理动机：MTF 反解 Gaussian PSF -> 面积降采样（= 像素孔径积分）-> 传感器噪声。
    综述 7.1 推荐的方向。sigma 由 MTF 反解，不手调。"""
    import cv2
    rng = rng or np.random.default_rng(0)
    sig = mtf_to_sigma(mtf)
    x = _gauss_blur(_as_float(img), sig)
    h, w = x.shape[:2]
    x = cv2.resize(x, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    x = add_sensor_noise(x, rng)
    return _to_uint8(x)


def deg_aniso(img, scale=4, rng=None, mtf=MTF_AT_NYQ, k_range=(2.0, 3.5)):
    """各向异性退化（v2）。

    v1 的做法是「先椭圆模糊 → 再面积降采样」，实测被下采样抹平
    （D4 vs D5 三个描述子全部 <0.5 sigma）。v2 改为**椭圆核直接充当降采样核**，
    方向性因此在分辨率下降后仍然存活。

    与 D4 的唯一差别：采样核的形状（椭圆 vs 盒），且几何平均 sigma 已对齐
    （s0 = 4/sqrt(12) ≈ 1.1547，与 INTER_AREA 等效），故测到的差异就是各向异性。
    """
    rng = rng or np.random.default_rng(0)
    x = _gauss_blur(_as_float(img), mtf_to_sigma(mtf))
    k = float(rng.uniform(*k_range))
    theta = float(rng.uniform(0, np.pi))
    s0 = 4.0 / np.sqrt(12.0)
    x = _aniso_resample(x, scale, k, theta, s0)
    x = add_sensor_noise(np.clip(x, 0.0, 1.0), rng)
    return _to_uint8(x)


def deg_aniso_iso_control(img, scale=4, rng=None, mtf=MTF_AT_NYQ):
    """★ D5 的**各向同性对照**：同一个椭圆重采样核，但 k=1（圆）。

    用途：D4(盒) vs D5(椭圆) 里混着"盒 vs 高斯"的差异。
    本对照把该差异单独提出来 ⇒
      D4 vs 本对照  = 「盒 vs 高斯」的贡献
      本对照 vs D5  = **纯各向异性**的贡献
    这样 D5 的效应才不会被归错因。
    """
    rng = rng or np.random.default_rng(0)
    x = _gauss_blur(_as_float(img), mtf_to_sigma(mtf))
    theta = float(rng.uniform(0, np.pi))
    s0 = 4.0 / np.sqrt(12.0)
    x = _aniso_resample(x, scale, 1.0, theta, s0)
    x = add_sensor_noise(np.clip(x, 0.0, 1.0), rng)
    return _to_uint8(x)


def arm_rng(image_id, arm, base=2026):
    """★ 逐图固定种子（v2 设计的核心约束）。

    同一 (图, 臂) 永远得到同一份随机序列，**与调用顺序/次数无关**。
    必须这样做的原因（两个漏洞是同一个根因）：
      (a) 否则 D2/D3 每次调用重新随机 ⇒ 偷带 data augmentation，混入第二个变量；
      (b) 否则同一张图两次退化不同 ⇒ 配对差分的 std(Δ) 里混的是采样方差，
          实测使配对提升从 6.2x 掉到 1.1x。
    """
    import zlib
    h = zlib.crc32(f"{base}|{image_id}|{arm}".encode("utf-8"))
    return np.random.default_rng(h)


def degrade(img, arm, scale=4, rng=None):
    if arm == "D0_interarea":
        return deg_interarea(img, scale)
    if arm == "D1_bicubic":
        return deg_bicubic(img, scale)
    if arm == "D2_highorder":
        return deg_highorder(img, scale, rng)
    if arm == "D3_shuffled":
        return deg_shuffled(img, scale, rng)
    if arm == "D4_physical":
        return deg_physical(img, scale, rng)
    if arm == "D5_aniso":
        return deg_aniso(img, scale, rng)
    raise ValueError(f"unknown arm: {arm}")


# ── M1 退化保真度指标（频域）────────────────────────────────────
def radial_spectrum_slope(img):
    """径向平均功率谱在 log-log 下的斜率。参考：自然图约 -2，越负越平滑。"""
    x = _as_float(img)
    if x.ndim == 3:
        x = x.mean(axis=2)
    x = x - x.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(x * win))
    P = (np.abs(F) ** 2).astype(np.float64)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = min(cy, cx) - 1
    prof = np.bincount(r.ravel(), P.ravel(), minlength=rmax + 1)[1:rmax + 1]
    cnt = np.bincount(r.ravel(), minlength=rmax + 1)[1:rmax + 1]
    prof = prof / np.maximum(cnt, 1)
    f = np.arange(1, len(prof) + 1)
    ok = prof > 0
    if ok.sum() < 8:
        return float("nan")
    a = np.polyfit(np.log(f[ok]), np.log(prof[ok]), 1)
    return float(a[0])


def noise_autocorr_len(img):
    """高频残差的水平自相关半高宽，**亚像素插值**（原版取整只有 1/2/3 三档，无分辨力）。"""
    x = _as_float(img)
    if x.ndim == 3:
        x = x.mean(axis=2)
    import cv2
    lo = cv2.GaussianBlur(x, (0, 0), 2.0)
    hf = x - lo
    hf = hf - hf.mean()
    # 行方向自相关，逐行算再平均（比 ravel 全图相关更干净）
    acs = []
    for r in hf:
        n = r.size
        v = r - r.mean()
        ac = np.correlate(v, v, mode="full")[n - 1:]
        if ac[0] > 1e-12:
            acs.append(ac / ac[0])
    if not acs:
        return float("nan")
    ac = np.mean(acs, axis=0)
    below = np.where(ac < 0.5)[0]
    if below.size == 0:
        return float(len(ac))
    i = int(below[0])
    if i == 0:
        return 0.0
    # 在 i-1 与 i 之间线性插值求 ac = 0.5 的位置
    y0, y1 = ac[i - 1], ac[i]
    frac = 0.0 if abs(y1 - y0) < 1e-12 else (y0 - 0.5) / (y0 - y1)
    return float((i - 1) + frac)


def spectral_anisotropy(img, band=(0.12, 0.45)):
    """谱的方向各向异性：在高频带内对频率方向做二阶矩张量，取 (L1-L2)/(L1+L2)。

    D5 的椭圆核会沿某一方向压低高频 ⇒ 径向平均会把这个信息完全抹掉，
    必须用二维方向统计才能看见。返回 (anisotropy, orientation_deg)。
    """
    x = _as_float(img)
    if x.ndim == 3:
        x = x.mean(axis=2)
    x = x - x.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(x * win))
    P = (np.abs(F) ** 2).astype(np.float64)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    fy = (yy - cy) / h
    fx = (xx - cx) / w
    r = np.sqrt(fx ** 2 + fy ** 2)
    m = (r > band[0]) & (r < band[1])
    if m.sum() < 50:
        return float("nan"), float("nan")
    ux, uy = fx[m], fy[m]
    nrm = np.sqrt(ux ** 2 + uy ** 2)
    ux, uy = ux / nrm, uy / nrm
    wgt = P[m] * (r[m] ** 2)          # 高频权重：低频谱斜率差异会淹没方向性
    s = wgt.sum()
    if s <= 0:
        return float("nan"), float("nan")
    Txx = float((wgt * ux * ux).sum() / s)
    Tyy = float((wgt * uy * uy).sum() / s)
    Txy = float((wgt * ux * uy).sum() / s)
    T = np.array([[Txx, Txy], [Txy, Tyy]])
    ev, evec = np.linalg.eigh(T)
    l1, l2 = float(ev[1]), float(ev[0])
    aniso = (l1 - l2) / (l1 + l2 + 1e-12)
    v = evec[:, 1]
    ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
    return float(aniso), ang


def fidelity_metrics(img):
    a, ang = spectral_anisotropy(img)
    return dict(spectral_slope=radial_spectrum_slope(img),
                noise_ac_len=noise_autocorr_len(img),
                anisot_high=a,
                orient_deg=ang)


# ── 自测 ────────────────────────────────────────────────────────
def _selftest():
    rng = np.random.default_rng(2026)
    h, w = 256, 256
    # 合成一张含小目标的测试图
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.35 + 0.15 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    for (cy, cx, s) in [(60, 70, 2), (140, 180, 3), (200, 40, 2)]:
        base += 0.45 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    img = _to_uint8(np.clip(base, 0, 1))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)

    sig = mtf_to_sigma()
    print(f"MTF 反解: MTF(0.5 cyc/px)={MTF_AT_NYQ}  ->  sigma={sig:.4f} px (原始像素)")
    print(f"          1D MTF 验证: exp(-2pi^2 s^2 f^2) at f=0.5 = "
          f"{np.exp(-2 * np.pi**2 * sig**2 * 0.25):.4f}")
    print()

    hdr = f"{'arm':<14}{'shape':<16}{'slope':>8}{'aclen':>8}{'aniso':>8}{'orient':>8}"
    print(hdr)
    print("-" * len(hdr))

    def row(tag, d):
        m = fidelity_metrics(d)
        print(f"{tag:<14}{str(d.shape):<16}{m['spectral_slope']:>8.3f}"
              f"{m['noise_ac_len']:>8.3f}{m['anisot_high']:>8.4f}"
              f"{m['orient_deg']:>8.1f}")

    row("HR(orig)", img)
    md = {}
    for arm in ARMS:
        d = degrade(img, arm, 4, np.random.default_rng(2026))
        md[arm] = fidelity_metrics(d)
        row(arm, d)

    print()
    print("=== 可分离性检查（原设计只查 slope；现补 aclen 与 aniso）===")
    for key in ("spectral_slope", "noise_ac_len", "anisot_high"):
        v = np.array([md[a][key] for a in ARMS], dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            print(f"  {key:<16} 无效")
            continue
        span = float(v.max() - v.min())
        print(f"  {key:<16} 跨臂极差 = {span:.4f}")

    print()
    print("=== 关键：D3 / D4 / D5 是否仍分不开（原设计的痛点）===")
    trio = ("D3_shuffled", "D4_physical", "D5_aniso")
    for key in ("spectral_slope", "noise_ac_len", "anisot_high"):
        vs = [md[a][key] for a in trio]
        span = max(vs) - min(vs)
        flag = "OK 已分开" if span > (0.05 if key != "noise_ac_len" else 0.05) else "!! 仍然重合"
        print(f"  {key:<16} D3/D4/D5 = "
              f"{vs[0]:.4f} / {vs[1]:.4f} / {vs[2]:.4f}   极差 {span:.4f}  {flag}")

    print()
    print("注：D5 的椭圆核方向随机(seed 2026)，其 orient_deg 应与该次核角度一致；")
    print("    各向异性 aniso 是径向平均看不见的那一维。")


if __name__ == "__main__":
    _selftest()
