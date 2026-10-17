"""SODA-A L2 —— 跨数据集复现的**第二个**数据集（4 臂 × 6 种子 × 16 条件）。

与 `dior_l2.py` 的关系（本文件的设计核心）
  **判决机器全部复用，只有"数据集侧"是新的。**
  * 条件网格 / 退化口径 / SR 引擎 / 训练超参 / 判据阈值 —— 逐字沿用
    `dior_l1.conditions()` 与 `dior_l2.TH`（同一套 `tv_sweep` 实现，不复制代码）。
  * 视图构建直接调 `dior_l1.build_view`（它本来就是"图像源可换"的），
    只是把 `L1.VIEW_ROOT` / `L1.DIOR_LBL` 指到 SODA 的瓦片与标签目录。
  * 判决段直接调 `dior_l2.verdict`（同一个函数、同一份判据、同一张分支表）。

数据集侧（`SODA_L2_PREREG.md`）
  瓦片来自 `soda_tile.py --crop` 的产物（策略 P7，规格见 §4）：
    `{TILE_ROOT}/train/images/{fname}.jpg`  ← 训练池 N_POOL_TRAIN = 2,804
    `{TILE_ROOT}/eval/images/{fname}.jpg`   ← 评测切片 500（官方 val）
  训练池按 `SODA_TILE_SEL.json` 的 md5 顺序切：
    前 2,103  → 训练（= AI-TOD 训练图数）
    后   701  → **训练期 val**（ultralytics 用它选 `best.pt`；不进任何判决）
  评测切片在评测时**现场退化 + SR**（与 DIOR/AI-TOD 同管线），不预建视图。

运行（WSL, 需 GPU；断点续跑）
  python soda_l2.py --prep                 # 只建视图
  python soda_l2.py                        # 全流程（默认 4 臂 × 6 种子）
  python soda_l2.py --verdict-only         # 只用已有 jsonl 出判决
"""
import argparse
import json
import os
import sys
import time

A2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/e/新方向研究/sr4sod")
sys.path.insert(0, A2)

import dior_l1 as L1          # ★ 视图 / 条件 / 常量（含 stdout 修补）
import dior_l2 as L2          # ★ 判决段（同一函数、同一判据）
import dcdc_scan as DC
import tv_sweep as TV

# ---------------------------------------------------------------- 数据集侧路径
SR4SOD_DATA = "/home/czy/data/sr4sod/soda_a"
TILE_ROOT = f"{SR4SOD_DATA}/tiles"
SEL = f"{SR4SOD_DATA}/SODA_TILE_SEL.json"
TRAIN_IMG = f"{TILE_ROOT}/train/images"
TRAIN_LBL = f"{TILE_ROOT}/train/labels"
EVAL_IMG = f"{TILE_ROOT}/eval/images"
EVAL_LBL = f"{TILE_ROOT}/eval/labels"

ROOT = f"{SR4SOD_DATA}/l2"
VIEW_ROOT = f"{ROOT}/views"
RUN_ROOT = f"{ROOT}/runs"
CK = os.path.join(A2, "soda_l2.jsonl")
OUT_J = os.path.join(A2, "soda_l2.json")
DONE = os.path.join(A2, "SODA_L2_DONE")

N_TRAIN = 2103                # 训练瓦片数（= AI-TOD 训练图数；`SODA_TILE_SEL.json` 前 2103 条）
N_POOL_TRAIN = 2804           # 训练池（= AI-TOD 的 POOL）；后 701 条当训练期 val

ARMS = L2.ARMS                # ★ 臂定义逐字沿用（A / B0 / B2 / D0 × blur/noise/jpeg）
SEEDS = L2.SEEDS
TB_MAIN = L1.TB_MAIN
TB_REPS = L1.TB_REPS
REF = L1.REF
HEAVY = L1.HEAVY


def say(m):
    print(m, flush=True)


def det_name(arm, seed):
    """★ 检测器名要带数据集前缀，否则 SODA 的缓存里会出现 `dior_l2_*` 这种误导名字。
    `dior_l2.verdict()` 在调用时按模块全局查 `det_name`，所以这里 patch 一下即可生效。"""
    return f"soda_l2_{arm}_s{seed}"


def load_sel():
    """读 `SODA_TILE_SEL.json`，按写入顺序（= md5 乱序）返回 (train_pool, eval)。"""
    if not os.path.isfile(SEL):
        raise SystemExit(f"★ 缺 {SEL} ⇒ 先跑 soda_tile.py --plan --crop")
    sel = json.load(open(SEL, encoding="utf-8"))
    tr = [r for r in sel if r.get("set") == "train"]
    ev = [r for r in sel if r.get("set") == "eval"]
    return tr, ev


def paths_of(rows, root):
    """瓦片 JSON 行 → 磁盘路径（裁瓦片时写的是 `<fname>.jpg`）。"""
    out, miss = [], []
    for r in rows:
        p = os.path.join(root, r["fname"] + ".jpg")
        if os.path.isfile(p):
            out.append(p)
        else:
            miss.append(r["fname"])
    return out, miss


def prep_data():
    """把 `dior_l1` / `dior_l2` 的机器指到 SODA 侧（**只改全局，不改任何函数体**）。"""
    os.makedirs(VIEW_ROOT, exist_ok=True)
    os.makedirs(RUN_ROOT, exist_ok=True)
    L1.VIEW_ROOT = VIEW_ROOT
    L1.DIOR_LBL = TRAIN_LBL          # ★ build_view 里标签是从这个全局拷的
    L2.A2 = ROOT                     # ★ 让 verdict 的 L1 漂移审计找不到 dior_l1.jsonl
    L2.DS = "SODA-A"                 # ★ 判决文本里的数据集名
    L2.det_name = det_name           # ★ 检测器名前缀
    L2.CK = CK                       # ★★ 缓存路径（否则 load_ck/append_ck 会去读 DIOR 的 jsonl）
    L2.RUN_ROOT = RUN_ROOT
    TV.RUN_ROOT = RUN_ROOT
    L1.CK = CK                       # 让 L1 的任何缓存调用也指向本文件的 jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(L2.CORE_ARMS))
    ap.add_argument("--seeds", default="")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"★ 未知臂 {a}；可选 {list(ARMS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s] or SEEDS
    n_tr = args.n_train

    # ---- 数据就绪检查（不靠状态文件，靠实际文件）----
    for d, what in ((TRAIN_IMG, "训练瓦片"), (TRAIN_LBL, "训练标签"),
                    (EVAL_IMG, "评测瓦片"), (EVAL_LBL, "评测标签")):
        if not os.path.isdir(d):
            raise SystemExit(f"★ 缺{what}目录 {d} ⇒ 先跑 soda_tile.py --crop")
    tr_rows, ev_rows = load_sel()
    if len(tr_rows) < n_tr:
        raise SystemExit(f"★ 训练池只有 {len(tr_rows)} 个瓦片（需要 ≥ {n_tr}）")
    files, miss_tr = paths_of(tr_rows, TRAIN_IMG)
    scan, miss_ev = paths_of(ev_rows, EVAL_IMG)
    if miss_tr or miss_ev:
        raise SystemExit(f"★ 瓦片文件缺失：训练 {len(miss_tr)} 个 / 评测 {len(miss_ev)} 个"
                         f" ⇒ 例 {miss_tr[:3] or miss_ev[:3]}")
    if n_tr >= len(files):
        raise SystemExit(f"★ --n-train {n_tr} 必须 < 训练池 {len(files)}（要留训练期 val）")

    say(f"[data] 训练池 {len(files):,} ⇒ 训练 {n_tr:,} / 训练期 val {len(files) - n_tr:,}"
        f"（`best.pt` 用后者选，不进判决）；评测切片 {len(scan):,}")
    say(f"[scope] 臂 {arms}；种子 {seeds}；"
        f"合计 {len(arms) * len(seeds)} 次训练 × {len(L1.conditions()) + len(TB_REPS)} 次评测")

    prep_data()
    done = L2.load_ck()
    say(f"[cache] 已有 {len(done)} 条 → {os.path.basename(CK)}")

    # ---- 阶段 1：视图 + 训练 + 评测（与 dior_l2.main 同构，只换数据与缓存路径）----
    if not args.verdict_only:
        eng = TV.make_engine("bsrgan")
        say(f"[engine] bsrgan ×{DC.SCALE}")
        ymls = {}
        for a in arms:
            cfg = ARMS[a]
            say(f"\n### 视图 {cfg['view']}  ({a}: {cfg['desc']}, blur σ ~ U{cfg['blur']})")
            # ★ n_tr < len(files) ⇒ build_view 自动把后 (len-n_tr) 张划进 images/val
            ymls[a] = L1.build_view(cfg["view"], cfg, eng, files, n_tr,
                                    cfg["noise"], cfg["jpeg"])
        if args.prep:
            say("\n[prep] 视图阶段结束（未训练）")
            return

        jobs = [(tg, tb, b, c) for (tg, b, c) in L1.conditions() for tb in [TB_MAIN]]
        jobs += [(L1.ref_suffix(), tb, REF[0], REF[1]) for tb in TB_REPS]
        say(f"\n[grid] 16 条件 + {len(TB_REPS)} 地板；ref tag = '{TB_MAIN}{L1.ref_suffix()}'")

        from ultralytics import YOLO
        for seed in seeds:
            say("\n" + "=" * 92)
            say(f"种子 {seed}")
            say("=" * 92)
            for a in arms:
                TV.train_one(ARMS[a]["view"], ymls[a], seed=seed, name=L2.det_name(a, seed))
            for a in arms:
                name = L2.det_name(a, seed)
                wp = os.path.join(RUN_ROOT, name, "weights", "best.pt")
                if not os.path.isfile(wp):
                    say(f"★ 缺权重 {wp}，跳过该臂")
                    continue
                m = YOLO(wp)
                for (sfx, tb, b, c) in jobs:
                    key = (name, tb + sfx)
                    if key in done:
                        continue
                    t0 = time.time()
                    r = DC.eval_arm(m, scan, EVAL_LBL, b, c, None, eng, tb + sfx, shot=False)
                    r.update(detector=name, tag=tb + sfx, blur=b, noise=c,
                             tag_base=tb, arm=a, seed=seed, shot=False, dataset="soda_a")
                    L2.append_ck(r)
                    done[key] = r
                    say(f"  {name}  {tb + sfx:<26} mAP5095="
                        f"{('%.4f' % r['mAP5095']) if r['mAP5095'] is not None else 'n/a'}"
                        f"  {time.time() - t0:.0f}s")
                del m

    # ---- 阶段 2：判决（**复用 `dior_l2.verdict`**，只换输出路径）----
    done = L2.load_ck()
    need = len(arms) * len(seeds) * (len(L1.conditions()) + len(TB_REPS))
    have = sum(1 for (d, t) in done if any(d == L2.det_name(a, s) for a in arms for s in seeds))
    say(f"\n[cache] 本设计 {have} 条 / 需要 {need} 条")
    if have < need:
        say("★ 评测不齐 —— 不写 SODA_L2_DONE，让监工续跑（防失败被静默掩盖）")
        return
    _saved = (L2.OUT_J, L2.DONE)
    L2.OUT_J, L2.DONE = OUT_J, DONE          # ★ 判决段只换这两个全局
    try:
        L2.verdict(done, arms, seeds, scan)
    finally:
        L2.OUT_J, L2.DONE = _saved


if __name__ == "__main__":
    main()
