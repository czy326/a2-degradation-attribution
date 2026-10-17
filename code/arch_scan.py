"""ARCH：第二个 nano 架构（YOLO11n）跑主数据集（AI-TOD）。

设计见 `ARCH_PREREG.md`（判据先写死在那里）。本脚本**不复制** `tv_sweep` 的任何逻辑，只做四件事：

  ① patch `tv_sweep.W_YOLO` → `yolo11n.pt`
  ② patch `tv_sweep.RUN_ROOT` → 独立目录（v8n 批的 run 名是 `tv_A_s2026`，本批用 `y11_` 前缀隔离）
  ③ 用 `tv_sweep` 自己的 CLI 驱动主批：4 臂 × 6 种子 × 16 条件（训练 + 评测），
     结果写 `tv_sweep.yolo11n.jsonl`
  ④ 追加**地板段**：每 (臂, 种子) 在 ref 条件 (blur=0.0, noise=0.003) 用 3 个 `tag_base`
     各评一次，写**独立文件** `tv_sweep.yolo11n.floor.jsonl`

★ 为什么地板要写独立文件：`tv_deploy.tagp()` 按 `tag.split("__")[1]` 解析条件，
  地板行若混进主 jsonl，会以同一个 (blur, noise) 键**覆盖**主结果里 ref 条件那一格。

用法
  python arch_scan.py            # 主批 + 地板（可断点续跑）
  python arch_scan.py --no-floor # 只跑主批
  python arch_scan.py --floor-only
"""
import glob
import io
import json
import os
import sys
import time

# ---- stdout 钉住：tv_sweep 会 import deg_ops，deg_ops 在 import 时替换 sys.stdout ----
_PINS = [sys.stdout, sys.stderr]

A2 = os.path.dirname(os.path.abspath(__file__))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)

import tv_sweep as TV          # noqa: E402  （复用，不复制）
import dcdc_scan as DC         # noqa: E402

_PINS.append(sys.stdout)
sys.stdout = _PINS[0]

W_YOLO11 = f"{SR4SOD}/_weights/yolo11n.pt"
RUN_ROOT11 = "/home/czy/data/sr4sod/tv11_runs"
CK_MAIN = os.path.join(A2, "tv_sweep.yolo11n.jsonl")
CK_FLOOR = os.path.join(A2, "tv_sweep.yolo11n.floor.jsonl")
DONE = os.path.join(A2, "ARCH_DONE")
FLOOR_DONE = os.path.join(A2, "ARCH_FLOOR_DONE")

ARMS = ["tv_A", "tv_B0", "tv_B2", "tv_D0"]
SEEDS = [2026, 2027, 2028, 2029, 2030, 2031]
# ★ eng-tag 取 "y11"（**不带下划线**）：run/检测器名 = `y11tv_A_s2026`。
#   ⚠ 2026-10-17 07:2x 更正：原来这里写「`tv_deploy` 里 `a1[3:]` 的截断刚好得到 `tv_A`，
#   臂标签与 v8n 批逐字相同」——这是**错的**：`a1[3:]("y11tv_A")` = `"tv_A"`，而 v8n 批
#   `a1[3:]("tv_A")` = `"A"`，两边并不同。已把 tv_deploy 的表头改成 `_sfx()`（去掉
#   `tv_` 之前的引擎前缀），两边都输出 `A` ⇒ 现在才真的逐字一致。
ENG_TAG = "y11"
SCAN_N = 200
REF = (0.0, 0.003)
TB_MAIN = "tvX_"
TB_REPS = ["tvY_", "tvZ_"]


def say(m):
    print(m, flush=True)


def setup():
    """把所有要改的模块全局集中在这里，便于自测逐条断言。"""
    if not os.path.isfile(W_YOLO11):
        raise SystemExit(f"★ 缺权重 {W_YOLO11}")
    TV.W_YOLO = W_YOLO11          # train_one 在调用时读这个全局
    TV.RUN_ROOT = RUN_ROOT11      # train_one / main 都读它
    # ★ 切断与 v8n 批缓存文件的不必要耦合：`tv_sweep.collect()` 会把 `PRELOAD`（默认
    #   `tv_sweep.jsonl`，即 v8n 批的 421 KB 结果）当成只读缓存读进来。当前键名不冲突
    #   （`tv_A_s2026` vs `y11tv_A_s2026`）因而无害，但那是**隐式**依赖；指到自己这份
    #   文件后，本批完全自洽（`CK` 由 collect 单独并入，见其 srcs 逻辑）。
    TV.PRELOAD = CK_MAIN
    os.makedirs(RUN_ROOT11, exist_ok=True)
    return dict(W_YOLO=TV.W_YOLO, RUN_ROOT=TV.RUN_ROOT, CK=TV.CK, VIEW_ROOT=TV.VIEW_ROOT,
                PRELOAD=TV.PRELOAD)


def run_main():
    argv = ["arch_scan.py",
            "--only", ",".join(ARMS),
            "--seeds", ",".join(str(s) for s in SEEDS),
            "--eng-tag", ENG_TAG,
            "--jsonl", CK_MAIN,
            "--scan-n", str(SCAN_N)]
    say(f"[arch] 主批 argv = {argv[1:]}")
    _saved = sys.argv
    sys.argv = argv
    try:
        TV.main()
    finally:
        sys.argv = _saved


def load_floor_done():
    got = set()
    if os.path.isfile(CK_FLOOR):
        for ln in open(CK_FLOOR, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    got.add((r["detector"], r["tag"]))
                except Exception:
                    pass
    return got


def run_floor():
    """每 (臂, 种子) 在 ref 条件用 3 个 tag_base 各评一次 ⇒ 退化实现噪声 SD_real。"""
    from ultralytics import YOLO
    files = sorted(glob.glob(os.path.join(TV.HR_IMG, "*.png")))[:TV.POOL]
    n_tr = int(round(len(files) * TV.TRAIN_FRAC))
    scan = files[n_tr:][:SCAN_N]
    eng = TV.make_engine("bsrgan")
    done = load_floor_done()
    say(f"\n[floor] 已有 {len(done)} 条 → {os.path.basename(CK_FLOOR)}")
    say(f"[floor] scan={len(scan)} 条件 ref={REF} tag_bases={[TB_MAIN] + TB_REPS}")
    for arm in ARMS:
        for sd in SEEDS:
            name = f"{ENG_TAG}{arm}_s{sd}"
            bp = os.path.join(RUN_ROOT11, name, "weights", "best.pt")
            if not os.path.isfile(bp):
                say(f"  [floor] {name}: 缺 best.pt，跳过")
                continue
            todo = [tb for tb in [TB_MAIN] + TB_REPS if (name, tb + "ref") not in done]
            if not todo:
                continue
            m = YOLO(bp)
            for tb in todo:
                tg = tb + "ref"
                t0 = time.time()
                r = DC.eval_arm(m, scan, TV.HR_LBL, REF[0], REF[1], None, eng,
                                tb + tg, shot=False, resample="area")
                r = dict(detector=name, arm=arm, seed=sd, tag_base=tb, tag=tg,
                         which="floor", mAP5095=float(r["mAP5095"]))
                with open(CK_FLOOR, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fh.flush()
                say(f"  [floor] {name:<18}{tb + tg:<12}mAP5095={r['mAP5095']:.4f}  "
                    f"{time.time() - t0:.0f}s")
            del m
    with open(FLOOR_DONE, "w") as fh:
        fh.write("ARCH_FLOOR_DONE\n")


def main():
    args = set(sys.argv[1:])
    if "--selftest" in args:
        selftest()
        return
    info = setup()
    say(f"[arch] 权重 = {info['W_YOLO']}")
    say(f"[arch] run 根 = {info['RUN_ROOT']}")
    say(f"[arch] 视图根 = {info['VIEW_ROOT']}（复用 v8n 批，不重建）")
    if "--floor-only" not in args:
        run_main()
        with open(DONE, "w") as fh:
            fh.write("ARCH_DONE\n")
    if "--no-floor" not in args:
        run_floor()


def selftest():
    """离线自测：断言每处 patch 生效 + 视图确实可复用（不训不评）。"""
    import io as _io
    out = _io.StringIO()
    info = setup()
    ok = True

    def chk(name, cond, got):
        nonlocal ok
        out.write(f"  [{'✔' if cond else '✘'}] {name}  got={got}\n")
        ok = ok and cond

    chk("TV.W_YOLO 指向 yolo11n", info["W_YOLO"].endswith("yolo11n.pt"), info["W_YOLO"])
    chk("权重文件存在", os.path.isfile(info["W_YOLO"]), os.path.getsize(info["W_YOLO"]))
    chk("TV.RUN_ROOT 已切到 tv11_runs", info["RUN_ROOT"].endswith("tv11_runs"), info["RUN_ROOT"])
    chk("TV.VIEW_ROOT 未动（复用 v8n 视图）",
        info["VIEW_ROOT"].endswith("tv_views"), info["VIEW_ROOT"])
    chk("TV.PRELOAD 已切到本批文件（不再把 v8n 结果当缓存读）",
        os.path.basename(info["PRELOAD"]) == "tv_sweep.yolo11n.jsonl",
        os.path.basename(info["PRELOAD"]))
    chk("CK_MAIN 与 v8n 的 tv_sweep.jsonl 不同",
        os.path.basename(CK_MAIN) != "tv_sweep.jsonl", os.path.basename(CK_MAIN))
    chk("CK_FLOOR 与 CK_MAIN 不同",
        os.path.abspath(CK_FLOOR) != os.path.abspath(CK_MAIN), os.path.basename(CK_FLOOR))
    # 视图可复用：4 臂都要有 2103 train / 701 val
    for a in ARMS:
        root = os.path.join(TV.VIEW_ROOT, a)
        it = len(os.listdir(os.path.join(root, "images", "train"))) if os.path.isdir(
            os.path.join(root, "images", "train")) else -1
        iv = len(os.listdir(os.path.join(root, "images", "val"))) if os.path.isdir(
            os.path.join(root, "images", "val")) else -1
        chk(f"视图 {a} 可复用 (train=2103, val=701)", it == 2103 and iv == 701, f"{it}/{iv}")
    chk("run 名带 y11 前缀（不与 v8n 冲突且臂标签可逐字对齐）",
        f"{ENG_TAG}{ARMS[0]}_s{SEEDS[0]}" == "y11tv_A_s2026",
        f"{ENG_TAG}{ARMS[0]}_s{SEEDS[0]}")
    argv = ["--only", ",".join(ARMS), "--seeds", ",".join(str(s) for s in SEEDS),
            "--eng-tag", ENG_TAG, "--jsonl", CK_MAIN, "--scan-n", str(SCAN_N)]
    chk("主批 argv 用独立 jsonl + y11 前缀",
        "--jsonl" in argv and CK_MAIN.endswith("tv_sweep.yolo11n.jsonl")
        and ENG_TAG == "y11", " ".join(argv))
    chk("地板 tag 不会被主 jsonl 承载（写独立文件）",
        os.path.abspath(CK_FLOOR) != os.path.abspath(CK_MAIN), os.path.basename(CK_FLOOR))
    out.write(f"\n  → {'全部通过' if ok else '★ 有失败项'}\n")
    print(out.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
