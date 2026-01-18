import os
import json
import argparse
import traceback

from drag_auto import run_drag_eval


def load_seed_points_jsonl(json_path, max_samples=None):
    """
    Load JSON Lines seed file:
    each line: {"seed": int, "points": {"handle": [...], "target": [...]}}

    Returns: list of dicts
    """
    records = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def run_freedragbench_eval(
    model_pkl,
    dataset_root,
    category,
    out_root,
    max_samples=None,
    drag_iterations=40,
    mask_type="center_circle",
    fixloss_type='single',
    tracking_method='L2',
    use_cascaded_blending=False
):
    """
    Run DragGAN evaluation on FreeDragBench-style dataset.
    """
    seed_file = os.path.join(dataset_root, "seeds", f"{category}.jsonl")
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"Seed file not found: {seed_file}")

    records = load_seed_points_jsonl(seed_file, max_samples=max_samples)

    print(f"[FreeDragBench] Category={category}")
    print(f"[FreeDragBench] Seed file={seed_file}")
    print(f"[FreeDragBench] Loaded {len(records)} samples")

    category_out = os.path.join(out_root, category)
    os.makedirs(category_out, exist_ok=True)

    for idx, rec in enumerate(records):
        seed = int(rec["seed"])
        handle_points = rec["points"]["handle"]
        target_points = rec["points"]["target"]

        sample_name = f"{idx:04d}_seed{seed}"
        sample_out = os.path.join(category_out, sample_name)
        os.makedirs(sample_out, exist_ok=True)

        print(f"\n=== [{idx+1}/{len(records)}] seed={seed}, pairs={len(handle_points)} ===")

        try:
            run_drag_eval(
                model_pkl=model_pkl,
                seed=seed,
                out_dir=sample_out,
                num_drag_points=len(handle_points),
                drag_iterations=drag_iterations,
                mask_type=mask_type,
                fixloss_type=fixloss_type,
                tracking_method=tracking_method,
                use_cascaded_blending=use_cascaded_blending,
                override_handle_points=handle_points,
                override_target_points=target_points,
            )
        except Exception as e:
            print(f"[ERROR] seed={seed} failed: {e}")
            traceback.print_exc()

    print("\n=== FreeDragBench Evaluation Finished ===")


def build_argparser():
    p = argparse.ArgumentParser("FreeDragBench batch evaluation for DragGAN")
    p.add_argument("--model", required=True, help="Path to generator model (.pkl)")
    p.add_argument("--dataset", required=True, help="FreeDragBench root folder (contains seeds/)")
    p.add_argument("--category", required=True, choices=["cars", "cats", "elephants", "faces", "horses"],
                   help="Which category seed file to run (seeds/<category>.json)")
    p.add_argument("--out", default="./eval_out_freedrag", help="Output root folder")
    p.add_argument("--max-samples", type=int, default=None, help="Max number of samples to run from jsonl")
    p.add_argument("--iterations", type=int, default=40, help="Drag optimization iterations")
    p.add_argument("--mask-type", type=str, default="center_circle",
                   choices=["center_circle", "center_rect", "all_ones"],
                   help="Mask type (kept same as your current script)")
    p.add_argument('--fixloss-type', type=str, default='single',
                        choices=['single', 'multilayer', 'raft','blended'],
                        help='Type of loss fixer')
    p.add_argument("--cascaded-blending", action="store_true",
                   help="Enable cascaded blending (same as your current script)")
    p.add_argument('--tracking-method', type=str, default='L2',
                        choices=['L2', 'mixed', 'raft','areawise'],
                        help='Type of loss fixer')
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    run_freedragbench_eval(
        model_pkl=args.model,
        dataset_root=args.dataset,
        category=args.category,
        out_root=args.out,
        max_samples=args.max_samples,
        drag_iterations=args.iterations,
        mask_type=args.mask_type,
        tracking_method=args.tracking_method,
        use_cascaded_blending=args.cascaded_blending
    )
