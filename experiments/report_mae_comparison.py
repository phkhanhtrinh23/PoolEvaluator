"""Produce the direct before/after-theory MAE table across all three domains.

The ESS/bound theory is a certificate and diagnostics layer; it does not change
the point estimator. Therefore the certificate-only post-theory MAE is expected to
equal the pre-theory PoolEval MAE. Keeping that row explicit prevents a theoretical
bound from being presented as an unrun estimator.

Example:
    python experiments/report_mae_comparison.py \
        --results-root /home/trinh/Dropbox/thesis/pool_text2sql_eval_code/results
"""
import argparse
import json
import os


def load_comparison(results_root):
    with open(os.path.join(results_root, "collision_formulation_real.json")) as handle:
        collision = json.load(handle)
    clean = ("spider", "sqlflow", "bird", "bird_minidev")
    text_before = sum(collision[name]["point"]["old"]["MAE"]
                      for name in clean) / len(clean)
    text_after_estimator = sum(collision[name]["point"]["corrected"]["MAE"]
                               for name in clean) / len(clean)
    with open(os.path.join(results_root, "domain_vision.json")) as handle:
        vision = json.load(handle)["summary"]["PoolEval"]["mae"] * 100.0
    with open(os.path.join(results_root, "domain_graph.json")) as handle:
        graph = json.load(handle)["summary"]["PoolEval"]["mae"] * 100.0

    before = {"Text2SQL": float(text_before), "Image Classification": float(vision),
              "Node Classification": float(graph)}
    after = dict(before)
    return {
        "units": "MAE percentage points",
        "before_theory": before,
        "after_theory_certificate_only": after,
        "change": {domain: 0.0 for domain in before},
        "after_theory_estimator_text2sql": float(text_after_estimator),
        "interpretation": (
            "The new theory supplies ESS and accuracy-bound diagnostics; it does "
            "not change the point estimator, so certificate-only MAE is unchanged."
        ),
    }


def main(results_root, out):
    result = load_comparison(results_root)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)

    print("MAE comparison (percentage points)")
    print(f"{'Method':40s} {'Text2SQL':>12s} {'Image':>12s} {'Node':>12s}")
    print("-" * 80)
    for label, values in [
        ("Before theory: PoolEval", result["before_theory"]),
        ("After theory: certificate only", result["after_theory_certificate_only"]),
        ("Change", result["change"]),
    ]:
        print(f"{label:40s} {values['Text2SQL']:12.2f} "
              f"{values['Image Classification']:12.2f} "
              f"{values['Node Classification']:12.2f}")
    print(f"{'After theory: ESS-anchored estimator':40s} "
          f"{result['after_theory_estimator_text2sql']:12.2f} {'N/A':>12s} {'N/A':>12s}")
    print("\nConclusion: the theory changes the certificate, not MAE, until an estimator "
          "uses the bound to alter its weights or prior.")
    print(f"[saved] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default="results/mae_comparison.json")
    main(**vars(parser.parse_args()))
