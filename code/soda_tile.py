"""SODA-A 切块（策略 P7）+ 训练/评测瓦片选定 —— 规格见 `SODA_L2_PREREG.md`。

两级解耦（沿用 DIOR 的教训：口径级的活不要被大文件下载阻塞）
  级 A  `--plan`  只读 55 MB 官方 JSON 标注 ⇒ 全量瓦片清单、逐桶框数、
                  选定 train 2,103 / eval 500，并**把 P7 的扫描数字逐项对上**
                  **零图像、零 GPU** —— 11 GB 图像还在下的时候就能跑完
  级 B  `--crop`  只裁**被选中的 3,304 个瓦片**（2,804 训练池 + 500 评测，不是全部 60,280 个），
                  写 YOLO 标签
        `--check` 裁完后自检：瓦片数 / 每瓦片标签行数 / 文件尺寸

★ P7 的确切定义（`SODA_L2_PREREG.md` §4，实现必须一模一样）
    对每张图 (W, H)：
      xs = [0, 800, 1600, …] 直到 ≤ W-800；若最后一块 ≠ W-800，追加 W-800
      ys = 同理
      瓦片 = xs × ys                        # 100% 覆盖
    对每个框 b（已排除 category_id=9）：
      找唯一一个满足 b 完整落在其中的瓦片；找不到 ⇒ 丢弃该框
      否则记到该瓦片，坐标原样保留（不做任何裁剪）
  本文件直接复用 `_soda_tile_scan.py` 的 `_tiles` / `inside`，避免两套实现漂移。

★ 口径：短边 = **先转 float32 再取 min(w,h)**（只在**报告**里用，不改框）。

用法
  python soda_tile.py --plan                 # 零图像，随时可跑
  python soda_tile.py --plan --limit 40      # 快速烟测（只读前 40 张）
  python soda_tile.py --crop                 # 需要 images/ 已解压
  python soda_tile.py --check
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import zipfile

import numpy as np

SR4SOD_E = "/mnt/e/新方向研究/sr4sod"
ANN_ZIP_L0 = f"{SR4SOD_E}/_data_cand/soda_a_annotations.zip"
CAND = "/home/czy/data/sr4sod/_data_cand/soda_a"
ANN_ZIP = f"{CAND}/Annotations.zip"
ROOT = "/home/czy/data/sr4sod/soda_a"
IMG_ROOT = f"{ROOT}/images"
TILE_ROOT = f"{ROOT}/tiles"
PLAN = f"{ROOT}/SODA_TILE_PLAN.json"
SEL = f"{ROOT}/SODA_TILE_SEL.json"

TILE = 800
IGNORE_ID = 9
EDGES = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 1e9]

# 预算（`SODA_L2_PREREG.md` §6，与 AI-TOD 对齐）
N_TRAIN = 2103          # 训练瓦片数 = AI-TOD 训练图数
N_EVAL = 500            # 评测切片瓦片数

# ★ 2026-10-16 22:0x 补的口径（记录见 `SODA_L2_PREREG.md` §6.1）
#   AI-TOD 是 `POOL=2804, TRAIN_FRAC=0.75` ⇒ **2,103 训练 + 701 留作训练期 val**
#   （ultralytics 用那 701 张选 `best.pt`；`tv_sweep.build_view` 里
#     `part = "train" if idx < n_tr else "val"`）。
#   `SODA_L2_PREREG.md` §6 只冻结了「训练 = 2,103」，**没写那 701 从哪来** ——
#   若不补，SODA 侧要么没有训练期 val（ultralytics 直接报错），要么把 2,103 再切 25%
#   ⇒ 训练集只剩 1,577，与 AI-TOD 的 2,103 不可比。
#   取 md5 乱序的**第 2,104–2,804 个** train 瓦片当 val ⇒ 前 2,103 个
#   （= `sel_train`）**逐字节不变**，§9 记录的选定集合完全不受影响。
N_POOL_TRAIN = 2804     # = AI-TOD 的 POOL

# ★ 审计基准：`_soda_tile.json` 的 P7 行（扫描时只用了标注）。
#   本脚本的第一件事就是**逐项复现**它；对不上就不许往下跑。
AUDIT = dict(
    n_tile=60_280,
    n_box=840_947,
    dropped=31_122,
    straddle=31_122,
    p50=11.87841796875,
    mid=0.6840419194075251,
    bins=[36, 8186, 49724, 83111, 288069, 154339, 154501, 54134, 48299, 548],
    per_split={"train": 25_613, "val": 13_816, "test": 20_851},
    # ★ 注意：扫描脚本里这个字段叫 `empty_tiles`，但它的实现是
    #     `tiles_per_img.append(len(bucket))` 后 `sum(1 for x in tiles_per_img if x == 0)`
    #   —— `bucket` 只装**收到过框**的瓦片，所以 `x == 0` 实际是
    #     **「一个框都没落到的图」的张数**，不是「空瓦片的个数」。
    #   本脚本按正确语义拆成两个量，只审计前者（=2）；后者（真空瓦片）另报。
    img_no_box=2,
)
EXPECT_KEEP = 872_069


def log(m=""):
    print(m, flush=True)


def pick_ann_zip():
    for p in (ANN_ZIP, ANN_ZIP_L0):
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------- P7 基本件
# ↓↓ 这两个函数与 `_soda_tile_scan.py` **逐字一致**（避免两套实现漂移）
def _tiles(W, H, stride, edge):
    xs = list(range(0, int(W) - TILE + 1, stride))
    ys = list(range(0, int(H) - TILE + 1, stride))
    if edge:
        if xs and xs[-1] != int(W) - TILE:
            xs.append(int(W) - TILE)
        if ys and ys[-1] != int(H) - TILE:
            ys.append(int(H) - TILE)
        if not xs and int(W) >= TILE:
            xs = [int(W) - TILE]
        if not ys and int(H) >= TILE:
            ys = [int(H) - TILE]
    return [(x, y) for y in ys for x in xs]


def inside(b, t):
    tx, ty = t
    return b[0] >= tx and b[1] >= ty and b[2] <= tx + TILE and b[3] <= ty + TILE
# ↑↑


def tiles_of_p7(W, H):
    return _tiles(W, H, TILE, True)


def short_sides_f32(boxes):
    """★ 先整体转 float32 再取短边。顺序不能换。"""
    if not boxes:
        return np.zeros(0)
    A = np.asarray(boxes, dtype=np.float32)
    return np.minimum(A[:, 2] - A[:, 0], A[:, 3] - A[:, 1]).astype(np.float64)


def size_bin(v):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= v < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def bins_of(ss):
    b = [0] * (len(EDGES) - 1)
    for v in ss:
        b[size_bin(float(v))] += 1
    return b


def short_of(boxes):
    """纯 python 版短边（float32 口径），逐框用。"""
    out = []
    for b in boxes:
        w = np.float32(b[2]) - np.float32(b[0])
        h = np.float32(b[3]) - np.float32(b[1])
        out.append(float(min(w, h)))
    return out


def parse_json(blob):
    """→ (boxes_keep, boxes_ignore, W, H, file_name)"""
    d = json.loads(blob)
    im = d.get("images") or {}
    if isinstance(im, list):
        im = im[0] if im else {}
    W, H = float(im.get("width") or 0), float(im.get("height") or 0)
    keep, ign = [], []
    if W > 0 and H > 0:
        for a in d.get("annotations") or []:
            poly = a.get("poly")
            if not poly or len(poly) < 8:
                continue
            xs, ys = poly[0::2], poly[1::2]
            b = (min(xs), min(ys), max(xs), max(ys))
            (ign if a.get("category_id") == IGNORE_ID else keep).append(b)
    return keep, ign, W, H, (im.get("file_name") or "")


def tid_of(split, stem, x, y):
    """瓦片标识（`SODA_L2_PREREG.md` §6，md5 的输入口径）。"""
    return f"{split}/{stem}__{x}__{y}"


def fname_of(split, stem, x, y):
    return f"{split}__{stem}__{x}__{y}"


def md5s(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 级 A
def plan(zp, limit=0, out=PLAN, sel_out=SEL):
    log("=" * 96)
    log(f"级 A  切块规划（P7）  标注 {zp}")
    log("=" * 96)
    zf = zipfile.ZipFile(zp)
    names = sorted(n for n in zf.namelist() if n.endswith(".json"))
    if limit:
        names = names[:limit]
        log(f"  ★ --limit {limit}：烟测模式，不做审计断言")

    # ---- pass 1：尺寸与瓦片标识（不碰框，内存极小）
    meta = []                                       # (split, stem, W, H, zip_name)
    tile_ids = collections.defaultdict(list)        # split -> [tid]
    for n in names:
        split = n.split("/", 1)[0]
        d = json.loads(zf.open(n).read())
        im = d.get("images") or {}
        if isinstance(im, list):
            im = im[0] if im else {}
        W, H = float(im.get("width") or 0), float(im.get("height") or 0)
        if W <= 0 or H <= 0:
            continue
        fn = im.get("file_name") or ""
        stem = os.path.splitext(os.path.basename(fn))[0] or os.path.splitext(
            os.path.basename(n))[0]
        meta.append((split, stem, W, H, n))
        for (x, y) in tiles_of_p7(W, H):
            tile_ids[split].append(tid_of(split, stem, x, y))

    n_tile = sum(len(v) for v in tile_ids.values())
    per_split_tile = {k: len(v) for k, v in tile_ids.items()}
    log(f"  有效图 {len(meta):,}；瓦片 {n_tile:,}；分 split 瓦片 {per_split_tile}")

    # ---- 选定：md5(瓦片标识) 确定性乱序取前 N
    def pick(split, n):
        ids = sorted(tile_ids[split], key=md5s)
        return ids[:n]

    sel_train = pick("train", N_TRAIN)
    sel_eval = pick("val", N_EVAL)
    # ★ 训练瓦片池 = md5 乱序前 N_POOL_TRAIN；前 N_TRAIN 个就是 `sel_train`（逐字节相同），
    #   第 N_TRAIN+1 … N_POOL_TRAIN 个留作**训练期 val**（见文件头 `N_POOL_TRAIN` 注释）。
    pool_train = pick("train", N_POOL_TRAIN)
    assert pool_train[:N_TRAIN] == sel_train, "★ 池前缀与 sel_train 不一致（md5 顺序被改动？）"
    selected = set(pool_train) | set(sel_eval)
    log(f"  选定：训练池 {len(pool_train):,} = 训练 {len(sel_train):,}"
        f" + 训练期 val {len(pool_train) - len(sel_train):,}（官方 train）"
        f"；评测 {len(sel_eval):,}（官方 val）")

    # ---- pass 2：框归属 + 全局审计
    per_tile_boxes = collections.defaultdict(list)
    n_kept = n_drop = n_straddle = n_ignore = 0
    ss_all = []
    empty = 0
    n_img_no_box = 0
    selected_src = collections.defaultdict(set)
    ign_center_in_eval = 0
    for split, stem, W, H, n in meta:
        keep, ign, W2, H2, fn = parse_json(zf.open(n).read())
        n_ignore += len(ign)
        tl = tiles_of_p7(W, H)
        got = collections.defaultdict(list)
        for b in keep:
            n_kept += 1
            hit = next((t for t in tl if inside(b, t)), None)
            if hit is None:
                n_straddle += 1
                n_drop += 1
                continue
            got[hit].append(b)
        if not got:
            n_img_no_box += 1
        for t in tl:
            bl = got.get(t, [])
            if not bl:
                empty += 1
            ss_all.extend(short_of(bl))
            tid = tid_of(split, stem, t[0], t[1])
            if tid in selected:
                per_tile_boxes[tid] = bl
                selected_src[tid.split("/", 1)[0]].add(stem)
                if split == "val":
                    for ib in ign:
                        cx, cy = (ib[0] + ib[2]) / 2.0, (ib[1] + ib[3]) / 2.0
                        if t[0] <= cx < t[0] + TILE and t[1] <= cy < t[1] + TILE:
                            ign_center_in_eval += 1

    ss = np.asarray(ss_all, dtype=np.float64)
    bins = bins_of(ss)
    p50 = float(np.median(ss))
    mid = float(((ss >= 4) & (ss < 16)).mean())
    log("")
    log("  审计（对照 `_soda_tile.json` 的 P7 行）：")
    rows = [
        ("瓦片数", n_tile, AUDIT["n_tile"]),
        ("框数", int(ss.size), AUDIT["n_box"]),
        ("丢弃（跨边界）", n_drop, AUDIT["dropped"]),
        ("中位短边", p50, AUDIT["p50"]),
        ("4–16 px 占比", mid, AUDIT["mid"]),
        ("无框图数", n_img_no_box, AUDIT["img_no_box"]),
    ]
    ok = True
    for k, got_v, exp_v in rows:
        same = (got_v == exp_v)
        ok &= same
        log(f"    {k:<16} 本次 {got_v!r:<22} 扫描 {exp_v!r:<22} {'✅' if same else '★ 不符'}")
    same = (bins == AUDIT["bins"])
    ok &= same
    log(f"    {'逐桶框数':<16} {'✅ 一致' if same else '★ 不符'}")
    if not same:
        log("      本次 " + "".join(f"{x:>7d}" for x in bins))
        log("      扫描 " + "".join(f"{x:>7d}" for x in AUDIT["bins"]))
    same = (per_split_tile == AUDIT["per_split"])
    ok &= same
    log(f"    {'分 split 瓦片':<16} {'✅ 一致' if same else '★ 不符'}")
    if not same:
        log(f"      本次 {per_split_tile}  扫描 {AUDIT['per_split']}")
    log(f"  去 ignore 框 {n_kept:,}（期望 {EXPECT_KEEP:,}）"
        f"；ignore 框 {n_ignore:,}；保留率 {ss.size / max(n_kept, 1):.4%}")
    log(f"  参考量：真空瓦片 {empty:,}/{n_tile:,}（{empty / n_tile:.2%}）"
        f" —— 扫描脚本没有正确报出这个量（见 `AUDIT` 上方注释），本行是新测的")

    # ★ 训练期 val 是**新加**的一块（见文件头 `N_POOL_TRAIN`），只影响 ultralytics 选 best.pt，
    #   不进任何判决；这里报出它的规模，供审计"它没把训练集吃掉"。
    val_ho = [v for tid in pool_train[N_TRAIN:] for v in short_of(per_tile_boxes.get(tid, []))]
    if val_ho:
        va = np.asarray(val_ho, dtype=np.float64)
        log(f"    （训练期 val {len(pool_train) - N_TRAIN:,} 瓦片：框 {va.size:,}；"
            f"框/瓦片 {va.size / max(len(pool_train) - N_TRAIN, 1):.2f}；"
            f"中位 {np.median(va):.3f} px） —— 只用于选 best.pt，不进判决")
    tr_ss = np.asarray([v for tid in sel_train for v in short_of(per_tile_boxes.get(tid, []))],
                       dtype=np.float64)
    ev_ss = np.asarray([v for tid in sel_eval for v in short_of(per_tile_boxes.get(tid, []))],
                       dtype=np.float64)
    log("")
    log("  选定集合的规模（不参与判决，作报告用）：")
    for nm, arr in (("训练", tr_ss), ("评测", ev_ss)):
        if arr.size == 0:
            log(f"    {nm}：0 框")
            continue
        log(f"    {nm}：框 {arr.size:,}；框/瓦片 {arr.size / (N_TRAIN if nm == '训练' else N_EVAL):.2f}；"
            f"中位 {np.median(arr):.3f} px；4–16 px {((arr >= 4) & (arr < 16)).mean():.2%}")
        log(f"          分桶 " + "".join(f"{x:>6d}" for x in bins_of(arr)))
    log(f"    覆盖源图 {len(selected_src['train']):,} 张（train 瓦片）"
        f" + {len(selected_src['val']):,} 张（eval 瓦片）"
        f" = {len(selected_src['train']) + len(selected_src['val']):,} 张"
        f"；官方 train 共 1,067 / val 共 576 张")
    log(f"    ★ 评测瓦片里 ignore 框中心落在瓦片内的数量 = {ign_center_in_eval:,}"
        f"（`SODA_L2_PREREG.md` §5.3 要求报告）")

    payload = dict(
        ann_zip=zp, policy="P7", tile=TILE, limit=limit,
        n_img=len(meta), n_tile=n_tile, per_split_tile=per_split_tile,
        n_box=int(ss.size), p50=p50, mid=mid, bins=bins,
        dropped=n_drop, straddle=n_straddle, empty_tiles=empty,
        img_no_box=n_img_no_box,
        n_train=N_TRAIN, n_eval=N_EVAL,
        n_pool_train=len(pool_train), n_val_holdout=len(pool_train) - len(sel_train),
        train_boxes=int(tr_ss.size), eval_boxes=int(ev_ss.size),
        n_src_img=len(selected_src["train"]) + len(selected_src["val"]),
        n_src_train=len(selected_src["train"]), n_src_eval=len(selected_src["val"]),
        ign_center_in_eval=ign_center_in_eval,
        audit_ok=bool(ok),
    )
    os.makedirs(ROOT, exist_ok=True)
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"  → {out}")

    sel = []
    for tid in pool_train:                       # ★ 池（含训练期 val）全裁
        split, rest = tid.split("/", 1)
        stem, x, y = rest.rsplit("__", 2)
        sel.append(dict(tid=tid, fname=fname_of(split, stem, int(x), int(y)),
                        split=split, stem=stem, x=int(x), y=int(y), set="train",
                        boxes=[[float(v) for v in b] for b in per_tile_boxes[tid]]))
    for tid in sel_eval:
        split, rest = tid.split("/", 1)
        stem, x, y = rest.rsplit("__", 2)
        sel.append(dict(tid=tid, fname=fname_of(split, stem, int(x), int(y)),
                        split=split, stem=stem, x=int(x), y=int(y), set="eval",
                        boxes=[[float(v) for v in b] for b in per_tile_boxes[tid]]))
    json.dump(sel, open(sel_out, "w", encoding="utf-8"))
    log(f"  → {sel_out}（{len(sel):,} 个瓦片的框清单）")
    if not limit and not ok:
        log("\n★ 审计未通过 ⇒ 不许往下裁瓦片")
        sys.exit(2)
    log("\n★ 级 A 通过" if not limit else "\n★ 级 A 烟测完成")
    return payload


# ---------------------------------------------------------------- 级 B
def crop(sel_path=SEL):
    from PIL import Image
    log("=" * 96)
    log(f"级 B  裁瓦片 + 写 YOLO 标签   {sel_path}")
    log("=" * 96)
    sel = json.load(open(sel_path, encoding="utf-8"))
    log(f"  待裁 {len(sel):,} 个瓦片")
    n_new = n_skip = 0
    miss = []
    for i, r in enumerate(sel, 1):
        src = find_image(r["stem"])
        if src is None:
            miss.append(r["fname"])
            continue
        x, y = r["x"], r["y"]
        outdir = os.path.join(TILE_ROOT, r["set"])
        os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "labels"), exist_ok=True)
        dst = os.path.join(outdir, "images", r["fname"] + ".jpg")
        lab = os.path.join(outdir, "labels", r["fname"] + ".txt")
        if os.path.isfile(dst) and os.path.isfile(lab):
            n_skip += 1
            continue
        with Image.open(src) as g:
            g = g.convert("RGB")
            t = g.crop((x, y, x + TILE, y + TILE))
            t.save(dst, quality=95, subsampling=0)
        lines = []
        for b in r["boxes"]:
            bx0 = min(max(b[0] - x, 0.0), TILE)
            by0 = min(max(b[1] - y, 0.0), TILE)
            bx1 = min(max(b[2] - x, 0.0), TILE)
            by1 = min(max(b[3] - y, 0.0), TILE)
            w, h = bx1 - bx0, by1 - by0
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {(bx0 + bx1) / 2 / TILE:.6f} {(by0 + by1) / 2 / TILE:.6f} "
                         f"{w / TILE:.6f} {h / TILE:.6f}")
        with open(lab, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_new += 1
        if i % 200 == 0:
            log(f"      {i:,}/{len(sel):,}")
    log(f"  ⇒ 新裁 {n_new:,}，已存在 {n_skip:,}")
    if miss:
        log(f"  ★ 找不到源图 {len(miss):,} 个 ⇒ 例：{miss[:5]}")
    return not miss


def find_image(stem):
    """SODA-A 的 Images.zip 是平铺的 00001.jpg（`_zip_peek.py` 实测）。"""
    for cand in (os.path.join(IMG_ROOT, stem + ".jpg"),
                 os.path.join(IMG_ROOT, stem + ".jpeg"),
                 os.path.join(IMG_ROOT, stem + ".png"),
                 os.path.join(IMG_ROOT, "images", stem + ".jpg")):
        if os.path.isfile(cand):
            return cand
    return None


def check(sel_path=SEL):
    from PIL import Image
    log("=" * 96)
    log(f"级 B 自检   {sel_path}")
    log("=" * 96)
    sel = json.load(open(sel_path, encoding="utf-8"))
    want = collections.Counter(r["set"] for r in sel)
    got = {}
    bad = 0
    tot_lab = 0
    tot_box = 0
    for st in ("train", "eval"):
        d = os.path.join(TILE_ROOT, st)
        imgs = sorted(os.listdir(os.path.join(d, "images"))) if os.path.isdir(
            os.path.join(d, "images")) else []
        got[st] = len(imgs)
        sizes = collections.Counter()
        for fn in imgs:
            p = os.path.join(d, "images", fn)
            with Image.open(p) as g:
                sizes[g.size] += 1
            lp = os.path.join(d, "labels", fn[:-4] + ".txt")
            if not os.path.isfile(lp):
                log(f"    ★ 缺标签 {fn}")
                bad += 1
                continue
            lines = [x for x in open(lp, encoding="utf-8").read().split("\n") if x.strip()]
            tot_lab += len(lines)
        tot_box += sum(len(r["boxes"]) for r in sel if r["set"] == st)
        log(f"    {st}: 瓦片 {got[st]:,}（期望 {want[st]:,}）；尺寸分布 "
            f"{dict((f'{a}x{b}', c) for (a, b), c in sizes.items())}")
    ok = (got.get("train") == want["train"] and got.get("eval") == want["eval"] and bad == 0)
    log(f"    标签行合计 {tot_lab:,}；规划框合计 {tot_box:,}"
        f"（差 = 退化框 {tot_box - tot_lab:,}）")
    log(f"  ★ 自检 {'通过' if ok else '★不通过★'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not (a.plan or a.crop or a.check):
        ap.print_help()
        return
    if a.plan:
        zp = pick_ann_zip()
        if zp is None:
            log("★ 找不到 SODA-A 标注包")
            sys.exit(1)
        # ★ 烟测模式**不许覆盖正式产物**（否则一次试跑就把权威清单冲掉）
        if a.limit:
            plan(zp, a.limit, out=f"{ROOT}/_SMOKE_plan.json",
                 sel_out=f"{ROOT}/_SMOKE_sel.json")
        else:
            plan(zp, 0, out=PLAN, sel_out=SEL)
    if a.crop:
        if not crop():
            sys.exit(3)
    if a.check:
        if not check():
            sys.exit(4)
    log("\n★ 本阶段全部通过")


if __name__ == "__main__":
    main()
