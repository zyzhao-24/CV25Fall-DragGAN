import os
import json
import argparse
import traceback
from typing import List, Dict, Optional, Any

from evaluation import run_evaluation


def is_sample_folder(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False

    fns = os.listdir(folder)
    has_img1 = any(fn.startswith("image1") and fn.endswith(".png") for fn in fns)
    has_img2 = any(fn.startswith("image2") and fn.endswith(".png") for fn in fns) or \
               any(fn.startswith("image2_blended") and fn.endswith(".png") for fn in fns)
    return has_img1 and has_img2


def find_sample_folders(root: str, recursive: bool = True) -> List[str]:
    samples = []
    if recursive:
        for dirpath, _, _ in os.walk(root):
            if is_sample_folder(dirpath):
                samples.append(dirpath)
    else:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if is_sample_folder(p):
                samples.append(p)

    samples.sort()
    return samples


def load_json_if_exists(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_valid_number(x: Any) -> bool:
    """Return True if x is a real finite number (not None, not NaN, not inf)."""
    try:
        if x is None:
            return False
        v = float(x)
        # NaN check: NaN != NaN
        if v != v:
            return False
        # inf check
        if v == float("inf") or v == float("-inf"):
            return False
        return True
    except Exception:
        return False


def _mean(xs: List[float]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return float(sum(xs) / len(xs))


def run_batch_eval(
    root: str,
    category: Optional[str],
    brighten_delta: int,
    thr: float,
    recursive: bool,
    save_summary_name: str = "summary.json",
):
    eval_root = os.path.join(root, category) if category else root
    if not os.path.exists(eval_root):
        raise FileNotFoundError(f"Eval root not found: {eval_root}")

    sample_folders = find_sample_folders(eval_root, recursive=recursive)
    print(f"[batch_eval] eval_root={eval_root}")
    print(f"[batch_eval] found {len(sample_folders)} sample folders")

    all_results: List[Dict] = []
    n_ok, n_fail = 0, 0

    for i, folder in enumerate(sample_folders):
        rel = os.path.relpath(folder, root)
        print(f"\n=== [{i+1}/{len(sample_folders)}] {rel} ===")

        try:
            res = run_evaluation(folder, brighten_delta=brighten_delta, thr=thr)
            if res is None:
                raise RuntimeError("run_evaluation returned None")
            n_ok += 1

            eval_results_path = os.path.join(folder, "eval_results.json")
            meta = load_json_if_exists(eval_results_path) or {}
            if "seed" in meta:
                res["seed"] = meta["seed"]
            if "drag_iterations" in meta:
                res["drag_iterations"] = meta["drag_iterations"]
            if "use_cascaded_blending" in meta:
                res["use_cascaded_blending"] = meta["use_cascaded_blending"]

            res["relative_folder"] = rel
            all_results.append(res)

        except Exception as e:
            n_fail += 1
            print(f"[ERROR] folder={folder} failed: {e}")
            traceback.print_exc()
            all_results.append({
                "relative_folder": rel,
                "folder": folder,
                "error": str(e),
            })

    # ===== 汇总统计：对 masked_l1_mean 和 lpips 求平均 =====
    masked_l1_vals: List[float] = []
    lpips_vals: List[float] = []

    for r in all_results:
        if not isinstance(r, dict):
            continue

        if "masked_l1_mean" in r and _is_valid_number(r["masked_l1_mean"]):
            masked_l1_vals.append(float(r["masked_l1_mean"]))

        if "lpips" in r and _is_valid_number(r["lpips"]):
            lpips_vals.append(float(r["lpips"]))

    summary = {
        "root": root,
        "category": category,
        "eval_root": eval_root,

        "num_samples": len(sample_folders),
        "num_ok": n_ok,
        "num_fail": n_fail,

        "brighten_delta": brighten_delta,
        "thr": thr,

        # masked L1 stats
        "masked_l1_mean_over_valid": _mean(masked_l1_vals),
        "masked_l1_num_valid": int(len(masked_l1_vals)),

        # LPIPS stats
        "lpips_mean_over_valid": _mean(lpips_vals),
        "lpips_num_valid": int(len(lpips_vals)),

        # per-sample results
        "results": all_results,
    }

    summary_path = os.path.join(eval_root, save_summary_name) if category else os.path.join(root, save_summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[batch_eval] Summary saved: {summary_path}")
    print(f"[batch_eval] OK={n_ok}, FAIL={n_fail}")
    print(f"[batch_eval] valid_masked_l1={len(masked_l1_vals)} mean={summary['masked_l1_mean_over_valid']}")
    print(f"[batch_eval] valid_lpips={len(lpips_vals)} mean={summary['lpips_mean_over_valid']}")
    return summary


def build_argparser():
    p = argparse.ArgumentParser("Batch evaluation for DragGAN outputs (diff + masked L1 + LPIPS mean)")
    p.add_argument("--root", required=True, help="Root folder of batch outputs, e.g. ./eval_out_freedrag")
    p.add_argument("--category", default=None,
                   choices=[None, "cars", "cats", "elephants", "faces", "horses", "lions"],
                   help="Optional: evaluate only one category subfolder")
    p.add_argument("--brighten-delta", type=int, default=25,
                   help="Brightness delta added to diff where mask < 0.5")
    p.add_argument("--thr", type=float, default=0.5,
                   help="Threshold for masked L1, compute over pixels where mask > thr")
    p.add_argument("--recursive", action="store_true",
                   help="Recursively find sample folders (recommended)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_batch_eval(
        root=args.root,
        category=args.category,
        brighten_delta=args.brighten_delta,
        thr=args.thr,
        recursive=args.recursive,
    )
