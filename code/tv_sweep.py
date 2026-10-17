"""TV 扫描 —— 退化不变性「维度特异性」判决

问题
  `model_dependence.py` 已建立相关性：检测器训练时的模糊范围越宽 ⇒ 对模糊越不敏感
  （r = -0.781，n = 6）。但相关性不能区分两种机制：

    H_spec（维度特异）  ：训练期见过宽模糊 ⇒ 只学到"抗模糊"，对噪声仍敏感
    H_gen （一般鲁棒）  ：训练期见过宽模糊 ⇒ 顺带变得对什么都钝

  两者对论文的含义完全不同：H_spec 说明「模型依赖」来自**该维度的不变性被训练出来**，
  是一个可预测、可控的机制；H_gen 说明它只是"训练分布越宽越钝"的平凡结论。

设计（2 个新检测器 + 2 个已存在的参照点，零破坏既有产物）
  训练端（唯一变量 = 训练期模糊范围）
    tv_B0 : blur σ ~ U(0.00, 0.05)   ← 几乎从不模糊
    tv_B2 : blur σ ~ U(0.20, 2.00)   ← 比 D2_highorder(0.2–1.2) 更宽
  参照：D0_interarea / D1_bicubic / HR_oracle (σ=0)、D2_highorder (0.2–1.2，已训)
  其余步骤（noise / JPEG / resize / epochs / imgsz / seed）**逐字保持 deg_highorder 原样**

  测量端（对每个检测器各测两条曲线）
    ① 模糊效应：blur 2.0 vs 0.2，噪声网格扫 PSNR ⇒ 匹配后配对
    ② 噪声效应：noise 0.096 vs 0.003，模糊网格扫 PSNR ⇒ 匹配后配对

判读
  |Δblur(tv_B2)| / |Δblur(tv_B0)| 显著 < 1  且
  |Δnoise(tv_B2)| / |Δnoise(tv_B0)| ≈ 1
  ⇒ H_spec 成立（维度特异）
  若两者同步下降 ⇒ H_gen

运行（WSL, 需 GPU）
  python tv_sweep.py --timing          # 只测一次 eval 的耗时
  python tv_sweep.py                   # 全跑（可断点续跑）
"""
import os
import sys
import io
import json
import time
import glob
import argparse
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)

# 注意：不要在这里包装 sys.stdout —— deg_ops 在 import 时已经包装过，
# 再次包装会让旧 wrapper 被 GC 回收并关掉底层 buffer（实测报 "I/O operation on closed file"）。

import deg_ops as D  # noqa: E402
from dcdc_scan import eval_arm, HR_IMG, HR_LBL, POOL, TRAIN_FRAC, BASE, LR, SCALE, save_json  # noqa: E402

RUNS_OLD = "/home/czy/data/sr4sod/a2_runs"
VIEW_ROOT = "/home/czy/data/sr4sod/tv_views"
RUN_ROOT = "/home/czy/data/sr4sod/tv_runs"
W_YOLO = f"{SR4SOD}/_weights/yolov8n.pt"
CK = os.path.join(A2, "tv_sweep.jsonl")
# 分片并行时的只读历史缓存（主结果文件）；分片新写的结果只进 CK
PRELOAD = os.path.join(A2, "tv_sweep.jsonl")

TRAIN_ARMS = {
    # ★ A 格：与 D0_interarea 等价的"无退化"训练（纯 INTER_AREA）。
    #   加它是为了让 2×2 的四个格都跑同一条管线、同一批种子 ⇒ 组间差才是干净的控制对比。
    "tv_A":  dict(blur=(0.00, 0.00), noise=False, jpeg=False, desc="无模糊 + 无噪声/JPEG"),
    "tv_B0": dict(blur=(0.00, 0.05), noise=True,  jpeg=True,  desc="窄模糊 + 噪声 + JPEG"),
    "tv_B2": dict(blur=(0.20, 2.00), noise=True,  jpeg=True,  desc="宽模糊 + 噪声 + JPEG"),
    # ★ 2×2 缺失的决定性格：宽模糊但**不含噪声/JPEG**。
    #   用来把「模糊范围」与「有没有噪声」这两个共线解释分开。
    "tv_D0": dict(blur=(0.20, 2.00), noise=False, jpeg=False, desc="宽模糊 + 无噪声/JPEG"),
    # ★★ 噪声 / JPEG 解耦（补齐**宽模糊层**的 2×2 因子格）。
    #   已有 tv_D0(宽+无) 与 tv_B2(宽+有) 只给出「噪声+JPEG 合起来」的效应，
    #   审稿人必问到底是哪一个在起作用。补这两格后 {D0, N2, J2, B2} 构成
    #   **blur 范围完全相同**（都是 U(0.20,2.00)）的 2×2 ⇒ 可直接算
    #   「噪声主效应 / JPEG 主效应 / 交互」，且**因子零混淆**。
    #   为什么不补窄模糊层：窄层的"A 格"blur 是 (0.00,0.00)、"B0 格"是 (0.00,0.05)，
    #   四格不在同一 blur 水平 ⇒ 拆出来的效应里会混进 0.00 vs 0.05 的模糊差；
    #   宽层四格 blur 严格一致，是唯一能做**无混淆** 2×2 的一层。
    "tv_N2": dict(blur=(0.20, 2.00), noise=True,  jpeg=False, desc="宽模糊 + 仅噪声（无 JPEG）"),
    "tv_J2": dict(blur=(0.20, 2.00), noise=False, jpeg=True,  desc="宽模糊 + 仅 JPEG（无噪声）"),
}

# 测量用的参照检测器（已存在，只补测噪声效应）
REF = {"D2_highorder": RUNS_OLD, "D0_interarea": RUNS_OLD, "tv_B0": RUN_ROOT, "tv_B2": RUN_ROOT}

# ① 模糊效应：固定 blur，扫噪声匹配 PSNR
BLUR_FAMS = {"blur2": 2.0, "blur02": 0.2}
NOISE_GRID = [0.003, 0.012, 0.024, 0.096]
# ② 噪声效应：固定 noise，扫模糊匹配 PSNR
NOISE_FAMS = {"nhi": 0.096, "nlo": 0.003}
BLUR_GRID = [0.0, 0.6, 1.2, 2.4]
# 默认只测两个新检测器（每个 16 次 eval ≈ 15 min）；加 --ref D2_highorder 可补参照点
REF = {}


# --------------------------------------------------------------------------
def degrade_span(img, scale, rng, blur_lo, blur_hi, use_noise=True, use_jpeg=True):
    """与 deg_ops.deg_highorder 逐字相同，差别仅在：模糊范围可配置、噪声/JPEG 可关。"""
    import cv2
    x = D._as_float(img)
    x = D._gauss_blur(x, float(rng.uniform(blur_lo, blur_hi)))
    h, w = x.shape[:2]
    x = cv2.resize(x, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    if use_noise:
        x = D.add_sensor_noise(x, rng)
    if use_jpeg:
        # ★★ 必须**直接返回** jpeg 的结果：jpeg_roundtrip 返回的是 uint8 [0,255]。
        #    早先这里在后面又套了一次 `_to_uint8(np.clip(x,0,1))`，
        #    会把 uint8[0,255] 先 clip 成 {0,1} 再 ×255 ⇒ **近白/二值图**（mean≈230~250）。
        #    该错误只在 use_jpeg=True 时触发（tv_A / tv_D0 不受影响）。
        #    已核验：四个已训臂的视图建于加开关之前，统计正常，历史结果不受污染。
        return D.jpeg_roundtrip(D._to_uint8(np.clip(x, 0, 1)), rng.integers(30, 96))
    return D._to_uint8(np.clip(x, 0, 1))


def build_view(tag, blur_lo, blur_hi, eng, pool=POOL, train_frac=TRAIN_FRAC,
               use_noise=True, use_jpeg=True, sub=""):
    """与 a2_pipeline.build_views 同构：退化 → LR → **BSRGAN 上采样回 800** → 落盘。
    上一版漏了 SR 这一步，会把 200px 图当训练图（已修）。"""
    import cv2
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    files = sorted(glob.glob(os.path.join(HR_IMG, "*.png")))[:pool]
    n_tr = max(1, int(round(len(files) * train_frac)))
    root = os.path.join(VIEW_ROOT, sub, tag) if sub else os.path.join(VIEW_ROOT, tag)
    for part in ("train", "val"):
        os.makedirs(os.path.join(root, "images", part), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", part), exist_ok=True)

    def one(idx_p):
        idx, p = idx_p
        stem = os.path.splitext(os.path.basename(p))[0]
        part = "train" if idx < n_tr else "val"
        dst = os.path.join(root, "images", part, stem + ".png")
        # ★ 断点续跑：已存在且**大小正常**（>=5 KB）才跳过。
        #   小于阈值的多半是上次被 WSL 回收时写到一半的残file。
        if os.path.isfile(dst) and os.path.getsize(dst) >= 5120:
            return 0
        hr = cv2.imread(p, cv2.IMREAD_COLOR)
        if hr is None:
            return 0
        if hr.shape[0] != BASE or hr.shape[1] != BASE:
            hr = cv2.resize(hr, (BASE, BASE), interpolation=cv2.INTER_AREA)
        lr = degrade_span(hr, SCALE, D.arm_rng(stem, tag), blur_lo, blur_hi,
                          use_noise=use_noise, use_jpeg=use_jpeg)
        if lr.shape[0] != LR or lr.shape[1] != LR:
            lr = cv2.resize(lr, (LR, LR), interpolation=cv2.INTER_AREA)
        # ★ 健康检查（改为**与源图比**的相对判据）
        #   教训：第一版用绝对阈值 mean>200，结果误伤了数据集里最亮的那张源图
        #   （13306d9ac 的 HR mean 本身就是 200.19）⇒ 绝对阈值不可用。
        #   正确的判据是"整幅图亮度不该被管线显著改变"：
        #     · 退化不该改亮度（LR vs HR）
        #     · SR 也不该改亮度（out vs LR）
        #   早先 `_to_uint8` 双重转换 bug 会把 HR 65 变成 230（+254%）⇒ 会被稳稳拦住。
        def _chk(a, b, who):
            tol = 0.25 * max(b, 1.0) + 12.0
            if abs(a - b) > tol:
                raise RuntimeError(
                    f"[view-sanity] {tag}/{stem} {who} 亮度异常：{a:.1f} vs 参照 {b:.1f} "
                    f"(容差 {tol:.1f}) —— 管线可能出错，拒绝写出")
        hm, lm = float(hr.mean()), float(lr.mean())
        _chk(lm, hm, "LR vs HR")
        out = eng.upscale(lr)                     # ★ SR 回 800，与 D0–D4 的条件对齐
        if out.shape[0] != BASE or out.shape[1] != BASE:
            out = cv2.resize(out, (BASE, BASE), interpolation=cv2.INTER_AREA)
        _chk(float(out.mean()), lm, "SR vs LR")
        cv2.imwrite(dst, out)
        lp = os.path.join(HR_LBL, stem + ".txt")
        if os.path.isfile(lp):
            shutil.copyfile(lp, os.path.join(root, "labels", part, stem + ".txt"))
        return 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as ex:      # SR 独占 GPU ⇒ 单线程
        n = sum(ex.map(one, list(enumerate(files))))
    yml = os.path.join(root, "data.yaml")
    with open(yml, "w", encoding="utf-8") as fh:
        fh.write(f"path: {root}\ntrain: images/train\nval: images/val\n")
        fh.write("nc: 1\nnames: ['object']\n")
    print(f"  [view] {tag}: {n} 张，{time.time()-t0:.0f}s → {root}", flush=True)
    return root, yml, n_tr


def _tcrit_note():
    return ""


class BicubicEngine:
    """无学习超分的对照引擎：直接 INTER_CUBIC ×4。

    为什么要它：假设「训练期模糊范围无效」是 BSRGAN 把模糊重构掉的产物。
    bicubic 不做任何复原 ⇒ LR 上的模糊会**原样传到检测器** ⇒ 这是对该假设最锐利的对照。
    """

    def __init__(self, scale=4):
        self.scale = scale
        self.device = "cpu"
        self.engine = "bicubic"

    def upscale(self, img):
        import cv2
        h, w = img.shape[:2]
        return cv2.resize(img, (w * self.scale, h * self.scale),
                          interpolation=cv2.INTER_CUBIC)


def make_engine(name, scale=4):
    if name == "bicubic":
        return BicubicEngine(scale)
    from probe0_sr import RealESRGAN
    return RealESRGAN(scale, engine=name)


def train_one(tag, yml, seed=2026, name=None):
    """★ 断点续跑的判定不能只看 best.pt —— YOLO 每轮验证变好就覆盖 best.pt，
    被 WSL 回收打断时留下的 best.pt 可能是**第 3 轮**的模型。
    因此必须同时确认 results.csv 已经写满 30 行（表头 + 30 epoch）。

    ★ 多种子：视图与**种子无关**（只建一次），run 目录按 `名字` 区分；
      名字默认 = tag，多种子时用 f"{tag}_s{seed}"。
    """
    from ultralytics import YOLO
    name = name or tag
    out = os.path.join(RUN_ROOT, name)
    bp = os.path.join(out, "weights", "best.pt")
    rc = os.path.join(out, "results.csv")
    n_ep = sum(1 for _ in open(rc)) if os.path.isfile(rc) else 0
    if os.path.isfile(bp) and n_ep > 30:
        print(f"  [train] {name}: 已完成（results.csv {n_ep} 行），跳过", flush=True)
        return bp
    if os.path.isdir(out):
        print(f"  [train] {name}: 残缺 run（results.csv {n_ep} 行）⇒ 重训，"
              f"避免误用早期 best.pt", flush=True)
    t0 = time.time()
    m = YOLO(W_YOLO)
    m.train(data=yml, epochs=30, imgsz=BASE, batch=8, seed=seed, deterministic=True,
            device=0, workers=8, project=RUN_ROOT, name=name, exist_ok=True,
            verbose=False, plots=False)
    print(f"  [train] {name}: {time.time()-t0:.0f}s", flush=True)
    return os.path.join(out, "weights", "best.pt")


# --------------------------------------------------------------------------
def paired_delta(rs, key="mAP5095", n_grid=7):
    cur = {}
    for fam in rs["_fams"]:
        pts = sorted((r["psnr"], r[key]) for r in rs["rows"]
                     if r["family"] == fam and r.get("psnr") and r.get(key) is not None)
        if len(pts) < 3:
            return None
        cur[fam] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    (fa, fb) = list(rs["_fams"])
    pa, ma = cur[fa]
    pb, mb = cur[fb]
    lo, hi = max(pa.min(), pb.min()), min(pa.max(), pb.max())
    if hi - lo < 0.3:
        return None
    grid = np.linspace(lo + 0.05, hi - 0.05, n_grid)
    d = np.interp(grid, pa, ma) - np.interp(grid, pb, mb)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return dict(mean=float(d.mean()), se=se,
                t=float(d.mean() / se) if se > 0 else float("nan"),
                n_neg=int((d < 0).sum()), n=len(d))


def collect(model, scan, eng, det, tag_base, which):
    """which='blur' 或 'noise'；返回该检测器的配对 Δ。"""
    fams = BLUR_FAMS if which == "blur" else NOISE_FAMS
    done = {}
    # ★ 缓存可来自两处：主文件（历史结果，只读）与本分片文件（本进程新写）
    srcs = []
    if os.path.isfile(PRELOAD) and os.path.abspath(PRELOAD) != os.path.abspath(CK):
        srcs.append(PRELOAD)
    if os.path.isfile(CK):
        srcs.append(CK)
    for src in srcs:
        for ln in open(src, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                done[(r["detector"], r["which"], r["tag"])] = r
            except Exception:
                pass
    if done:
        print(f"    [cache] {det}/{which}: 命中 {len(done)} 条", flush=True)
    rows = []
    if which == "blur":
        pairs = [(f, b, c) for f, b in fams.items() for c in NOISE_GRID]
        for f, b, c in pairs:
            tg = f"{f}__b{b:g}_n{c:g}"
            k = (det, which, tg)
            if k in done:
                rows.append(done[k])
                continue
            r = eval_arm(model, scan, HR_LBL, b, c, None, eng, tag_base + tg, shot=False)
            r.update(detector=det, which=which, family=f, tag=tg, shot=False)
            _append(r)
            rows.append(r)
    else:
        pairs = [(f, b, c) for f, c in fams.items() for b in BLUR_GRID]
        for f, b, c in pairs:
            tg = f"{f}__n{c:g}_b{b:g}"
            k = (det, which, tg)
            if k in done:
                rows.append(done[k])
                continue
            r = eval_arm(model, scan, HR_LBL, b, c, None, eng, tag_base + tg, shot=False)
            r.update(detector=det, which=which, family=f, tag=tg, shot=False)
            _append(r)
            rows.append(r)
    return paired_delta(dict(rows=rows, _fams=list(fams.keys())))


def _append(r):
    with open(CK, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--scan-n", type=int, default=200)
    ap.add_argument("--only", default="")
    ap.add_argument("--ref", default="", help="逗号分隔，额外测量这些已存在的检测器（a2_runs 下）")
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--smoke", action="store_true", help="只建 40 图的小视图并退出")
    # ★ 并行支持（评测是扩展种子数时的真正瓶颈）
    ap.add_argument("--no-train", action="store_true", help="跳过视图/训练，只做评测")
    ap.add_argument("--train-only", action="store_true", help="只做视图/训练，不评测")
    ap.add_argument("--seeds", default="2026",
                    help="逗号分隔的训练种子。★ 视图与种子无关（只建一次）；"
                         "多种子时 run 目录为 {arm}_s{seed}，"
                         "单种子(2026)时保持旧命名 {arm} 以复用已有权重")
    # ★ 超分引擎轴：检验「训练期模糊范围无效」是否引擎特异
    ap.add_argument("--engine", default="bsrgan",
                    choices=["bsrgan", "realesrgan", "bicubic"],
                    help="视图与评测用的超分引擎。bsrgan 为默认（与已有 6 种子批次一致）；"
                         "bicubic = 无学习超分对照（不做复原 ⇒ 模糊原样传到检测器）")
    ap.add_argument("--eng-tag", default="",
                    help="run/检测器名的前缀；留空时 bsrgan 不加前缀（复用已有），其余自动加")
    ap.add_argument("--shard", default="0/1", help="K/N：只处理第 K 个分片（从 0 起）")
    ap.add_argument("--jsonl", default="", help="结果 JSON Lines 路径（并行时每分片一份；"
                                               "★ /mnt/e 是 drvfs，并发 append 会静默丢行，"
                                               "所以不能让多个分片写同一个文件）")
    args = ap.parse_args()

    global CK, PRELOAD
    if args.jsonl:
        CK = args.jsonl if os.path.isabs(args.jsonl) else os.path.join(A2, args.jsonl)
        print(f"[shard] 结果文件 → {CK}")
    # ★ 引擎实验的「只读历史缓存」必须指向**本引擎**的主结果文件。
    #   原先写死 PRELOAD = tv_sweep.jsonl（bsrgan 的 4 臂 × 6 种子），其 detector 名不带
    #   引擎前缀，与引擎实验的键名（bicubic_tv_A_s2026…）永不匹配 ⇒ 后果有两个：
    #   ① 补种子时，旧的 3 种子结果命中不了缓存 ⇒ 白重评 96 条；
    #   ② 更危险：tv_engine_exp.sh 的 merge_eng 把分片并入主文件后**删除分片**，
    #      下一轮分片是新文件、缓存为空 ⇒ 监工**每轮都全量重评**（96→192 条），
    #      多轮重试会放大成数小时浪费。
    if args.engine != "bsrgan":
        _em = os.path.join(A2, f"tv_sweep.{args.engine}.jsonl")
        if os.path.isfile(_em):
            PRELOAD = _em
            print(f"[cache] 只读历史缓存 → {os.path.basename(PRELOAD)}")

    os.makedirs(VIEW_ROOT, exist_ok=True)
    os.makedirs(RUN_ROOT, exist_ok=True)

    files = sorted(glob.glob(os.path.join(HR_IMG, "*.png")))[:POOL]
    n_tr = int(round(len(files) * TRAIN_FRAC))
    scan = files[n_tr:][:args.scan_n]
    print(f"pool={len(files)} train={n_tr} scan={len(scan)}")

    from ultralytics import YOLO
    eng = make_engine(args.engine)
    print(f"[engine] {args.engine}")

    if args.timing:
        m = YOLO(f"{RUNS_OLD}/a2_HR_oracle/weights/best.pt")
        t0 = time.time()
        eval_arm(m, scan[:20], HR_LBL, 2.0, 0.012, None, eng, "tv_timing", shot=False)
        print(f"eval_arm 20 图：{time.time()-t0:.1f}s ⇒ 200 图约 {(time.time()-t0)*10:.0f}s")
        return

    only = [s for s in args.only.split(",") if s] or list(TRAIN_ARMS)
    SEEDS = [int(x) for x in args.seeds.split(",") if x.strip()]
    RUNNAME = (lambda tag, sd: tag) if SEEDS == [2026] else (lambda tag, sd: f"{tag}_s{sd}")
    # ★ 引擎前缀：bsrgan 不加（复用已有 6 种子权重）；其余引擎加前缀以隔离命名
    engtag = args.eng_tag or ("" if args.engine == "bsrgan" else f"{args.engine}_")
    sub = "" if args.engine == "bsrgan" else args.engine
    print(f"[seeds] {SEEDS}   run 命名：{engtag}{RUNNAME('tv_A', SEEDS[0])}   "
          f"视图子目录：{sub or '(根)'}")
    if args.smoke:
        tag = only[0]
        cfg = TRAIN_ARMS[tag]
        print(f"[smoke] 建 40 图视图验证管线：{tag} blur σ ~ U{cfg['blur']}", flush=True)
        # ★ 必须把该臂的 noise/jpeg 开关一起传进去：
        #   否则默认 (True, True)，tv_N2/tv_J2 的「仅噪声 / 仅 JPEG」两条新路径
        #   **根本不会被 smoke 走到** —— 审计就成了空转。
        root, yml, ntr = build_view(tag + "_smoke", *cfg["blur"], eng, pool=40, sub=sub,
                                    use_noise=cfg.get("noise", True),
                                    use_jpeg=cfg.get("jpeg", True))
        print(f"[smoke] OK → {root}, n_train={ntr}", flush=True)
        import glob as _g
        fs = sorted(_g.glob(os.path.join(root, "images", "train", "*.png")))[:2]
        for f in fs:
            import cv2
            im = cv2.imread(f)
            print(f"  {os.path.basename(f)}: shape={None if im is None else im.shape}", flush=True)
        return

    for tag in only:
        if args.no_train:
            break
        cfg = TRAIN_ARMS[tag]
        print(f"\n### {tag}  ({cfg['desc']}, blur σ ~ U{cfg['blur']})", flush=True)
        # ★ 视图只建一次（与种子无关）
        root, yml, _ = build_view(tag, *cfg["blur"], eng, pool=args.pool,
                                  use_noise=cfg.get("noise", True),
                                  use_jpeg=cfg.get("jpeg", True), sub=sub)
        for sd in SEEDS:
            train_one(tag, yml, seed=sd, name=engtag + RUNNAME(tag, sd))

    if args.train_only:
        print("\n[train-only] 视图/训练阶段结束，未做评测", flush=True)
        return

    ref = {}
    for tag in only:
        for sd in SEEDS:
            ref[engtag + RUNNAME(tag, sd)] = RUN_ROOT
    for t in [s for s in args.ref.split(",") if s]:
        ref[t] = RUNS_OLD
    # ★ 分片：只在评测阶段切分（视图/训练与种子无关，必须只做一次）
    try:
        K, NS = (int(x) for x in args.shard.split("/"))
    except Exception:
        raise SystemExit(f"--shard 格式应为 K/N，收到 {args.shard!r}")
    if NS > 1:
        all_dets = list(ref)
        keep = [d for i, d in enumerate(all_dets) if i % NS == K]
        ref = {d: ref[d] for d in keep}
        print(f"[shard] {K}/{NS} → 本片负责 {keep}", flush=True)
    print(f"\n测量对象：{list(ref)}", flush=True)

    print("\n" + "=" * 96)
    print("测量两条曲线（模糊效应 / 噪声效应），PSNR 匹配后配对")
    print("=" * 96)
    res = {}
    rows = []
    print(f"{'detector':<16}{'|Δblur|':>12}{'t':>8}{'|Δnoise|':>12}{'t':>8}", flush=True)
    for det, base in ref.items():
        wp = os.path.join(base, f"{det}", "weights", "best.pt")
        if not os.path.isfile(wp):
            print(f"{det:<16}  (缺 best.pt: {wp})", flush=True)
            continue
        m = YOLO(wp)
        # ★ 固定的 tag_base（**不含检测器名**）：让所有检测器看到**完全相同**的退化图，
        #   于是检测器之间的对比变成**逐图配对**，大幅提高组间检验的功效。
        #   （此前用 f"tvX_{det}_" 是错的 —— 每个检测器看到不同的退化实现。
        #    这也与 blur_indep_cross.jsonl 的口径一致：那边所有检测器共用同一 tag。）
        tb0 = "tvX_"
        sb = collect(m, scan, eng, det, tb0, "blur")
        sn = collect(m, scan, eng, det, tb0, "noise")
        del m
        res[det] = dict(blur=sb, noise=sn)
        fb = f"{sb['mean']:>+12.5f}" if sb else f"{'n/a':>12}"
        fn = f"{sn['mean']:>+12.5f}" if sn else f"{'n/a':>12}"
        t1 = f"{sb['t']:>8.2f}" if sb else f"{'':>8}"
        t2 = f"{sn['t']:>8.2f}" if sn else f"{'':>8}"
        print(f"{det:<16}{fb}{t1}{fn}{t2}", flush=True)
        rows.append((det, sb, sn))
        save_json(os.path.join(A2, (f"tv_result_{engtag.strip('_')}.json" if engtag
                                    else ("tv_result.json" if NS == 1
                                          else f"tv_result_shard{K}.json"))), res)

    # ---- 判读 ----
    print("\n" + "=" * 96)
    print("判读：H_spec（维度特异） vs H_gen（一般鲁棒）")
    print("=" * 96)
    def ab(d, k):
        v = res.get(d, {}).get(k)
        return abs(v["mean"]) if v else None
    a0, a2 = ab("tv_B0", "blur"), ab("tv_B2", "blur")
    n0, n2 = ab("tv_B0", "noise"), ab("tv_B2", "noise")
    if None in (a0, a2, n0, n2):
        print("  数据不全，无法判读（可能某个检测器重叠 PSNR 区间不足）")
    else:
        rb, rn = a2 / a0, n2 / n0
        print(f"  |Δblur|  : tv_B0 = {a0:.5f}  →  tv_B2 = {a2:.5f}   ⇒ 比值 {rb:.2f}")
        print(f"  |Δnoise| : tv_B0 = {n0:.5f}  →  tv_B2 = {n2:.5f}   ⇒ 比值 {rn:.2f}")
        print()
        if rb < 0.75 and 0.6 < rn < 1.6:
            print("  ⇒ ★ **H_spec 成立**：宽模糊训练只钝化了『模糊』这一个维度，噪声敏感性基本不变。")
            print("     ⇒ 「模型依赖」是**维度特异的不变性**被训练出来的结果，可预测、可控制。")
        elif rb < 0.75 and rn < 0.75:
            print("  ⇒ **H_gen 成立**：宽模糊训练同时钝化了模糊与噪声 ⇒ 只是一般鲁棒性。")
        else:
            print(f"  ⇒ 未判死也未判活（rb={rb:.2f}, rn={rn:.2f}）——需补第三个臂或更多种子。")

    save_json(os.path.join(A2, (f"tv_result_{engtag.strip('_')}.json" if engtag
                                    else ("tv_result.json" if NS == 1
                                          else f"tv_result_shard{K}.json"))), res)
    # ★ 完成标记：只看**模糊曲线**（插值法能可靠算出它）。
    #   噪声曲线必须用「固定模糊逐点配对」算，见 tv_analyze.py —— 插值法在本管线上无效
    #   （同噪声下 blur 0→2.4 只让 PSNR 动 0.10 dB，两族毫无重叠）。
    # ★ 必须用 only + engtag：否则引擎实验（名字带前缀、只训 2 臂）与 bsrgan 的 4 臂名字永不匹配
    need = {f"{engtag}{RUNNAME(t, sd)}|blur" for t in only for sd in SEEDS}
    have = {f"{d}|blur" for d in ref if res.get(d, {}).get("blur")}
    if NS > 1:
        print(f"\n[shard] {K}/{NS} 本片完成；TV_DONE 由 tv_parallel.sh 的合并步骤写")
    elif need <= have:
        with open(os.path.join(A2, "TV_DONE"), "w") as fh:
            fh.write("ok\n")
        print("\n★ TV_DONE 已写：所有臂的模糊曲线齐了（噪声曲线请跑 tv_analyze.py）")
    else:
        print(f"\n⚠ 未写 TV_DONE：缺 {sorted(need - have)}")
    print(f"\nwrote {os.path.basename(CK)}")


if __name__ == "__main__":
    main()
