"""Per-model prior gap tables: labeled accuracy vs true target accuracy, per modality.

One table per modality, both datasets side by side, one row per model:

    labeled  accuracy on the labeled split -- what the prior claims
    target   true accuracy on the unlabeled target pool -- what we want
    error    labeled - target, signed, so the direction of the mistake is visible

The signed column is the point.  If every entry carries the same sign the prior is not
noisy, it is systematically mis-levelled, and the summary row separates the two:

    bias  mean signed error -- the shared level offset
    MAE   mean |error| -- contains that offset
    cMAE  mean |error - bias| -- what remains once the offset is removed, i.e. whether
          the prior still ranks the models correctly

Writes a box-drawn table to stdout and a markdown copy to docs/prior_gap_tables.md.

  python experiments/report_prior_gap.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")
OUT = os.path.join(ROOT, "docs", "prior_gap_tables.md")

MODALITIES = [
    ("Text-to-SQL", [("spider", "Spider"), ("bird", "BIRD")]),
    ("Image classification", [("vision_mnist_usps", "MNIST->USPS"),
                              ("vision_mnist_svhn", "MNIST->SVHN")]),
    ("Node classification", [("graph_AC", "A->C"), ("graph_DA", "D->A")]),
]


def text2sql_case(ds):
    from zoo.new_formulation_real import load_run
    run, names = load_run(ds)
    src = np.load(os.path.join(ROOT, "zoo_artifacts", "source_true_class.npz"))
    return ([str(n) for n in names],
            (src[ds] == 0).mean(axis=1) * 100.0,
            np.asarray(run.true_acc) * 100.0)


def pool_case(key):
    z = np.load(os.path.join(POOL_CACHE, key + ".npz"), allow_pickle=True)
    lab = (z["prob_s"].argmax(-1) == z["src_gold_val"][None, :]).mean(axis=1) * 100.0
    tgt = (z["pred"] == z["gold"][None, :]).mean(axis=1) * 100.0
    return [str(n) for n in z["names"]], lab, tgt


def stats(lab, tgt):
    d = lab - tgt
    bias = float(d.mean())
    return bias, float(np.abs(d).mean()), float(np.abs(d - bias).mean())


def box(headers, rows, rule_after=None):
    """Box-drawn table with a rule between every row, matching the requested style."""
    w = [max(len(headers[c]), max((len(r[c]) for r in rows), default=0)) + 2
         for c in range(len(headers))]
    def rule(l, m, r, fill="─"):
        return l + m.join(fill * x for x in w) + r
    fmt = lambda cells: "│" + "│".join(
        " " + c.ljust(x - 2) + " " for c, x in zip(cells, w)) + "│"
    out = [rule("┌", "┬", "┐"), fmt(headers)]
    for i, r in enumerate(rows):
        # A heavy rule separates the per-model rows from the bias/MAE/cMAE summary.
        if rule_after is not None and i == rule_after + 1:
            out.append(rule("╞", "╪", "╡", fill="═"))
        else:
            out.append(rule("├", "┼", "┤"))
        out.append(fmt(r))
    out.append(rule("└", "┴", "┘"))
    return "\n".join(out)


def build(modality, cases):
    loader = text2sql_case if modality == "Text-to-SQL" else pool_case
    data = [(label,) + loader(key) for key, label in cases]
    names = data[0][1]
    headers = ["model"]
    for label, _, _, _ in data:
        headers += [f"labeled ({label})", f"target ({label})", "error"]
    rows = []
    for i, n in enumerate(names):
        r = [n]
        for _, _, lab, tgt in data:
            r += [f"{lab[i]:.1f}", f"{tgt[i]:.1f}", f"{lab[i] - tgt[i]:+.1f}"]
        rows.append(r)
    n_models = len(rows)
    for tag, idx in [("bias", 0), ("MAE", 1), ("cMAE", 2)]:
        r = [tag]
        for _, _, lab, tgt in data:
            v = stats(lab, tgt)[idx]
            r += ["", "", (f"{v:+.2f}" if tag == "bias" else f"{v:.2f}")]
        rows.append(r)
    return headers, rows, n_models - 1


def main():
    md = ["# Prior gap, model by model",
          "",
          "For each model two accuracies, in **accuracy percentage points**:",
          "",
          "- **labeled** -- its accuracy on the labeled split (what the prior tells us)",
          "- **target** -- its true accuracy on the unlabeled target pool (what we "
          "actually want)",
          "",
          "and the signed gap between them, `error = labeled - target`. A positive error "
          "means the prior is *too optimistic* about that model.",
          "",
          "The three summary rows separate the two kinds of mistake a prior can make:",
          "",
          "| row | formula | question it answers |",
          "|---|---|---|",
          "| `bias` | mean of the signed errors | is the prior wrong in the same "
          "direction for every model? |",
          "| `MAE` | mean of `|error|` | how wrong is the prior overall? (this number "
          "*contains* the bias) |",
          "| `cMAE` | mean of `|error - bias|` | once the shared offset is removed, does "
          "the prior still rank the models correctly? |",
          "",
          "`bias == MAE` means every single model is overstated, by nearly the same "
          "amount -- a pure **level** error, which a handful of target labels can "
          "correct. A large `cMAE` means a **shape** error, which they cannot.",
          ""]
    for modality, cases in MODALITIES:
        headers, rows, rule = build(modality, cases)
        print(f"\n## {modality}\n")
        print(box(headers, rows, rule_after=rule))
        md += [f"## {modality}", "",
               "| " + " | ".join(headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        for i, r in enumerate(rows):
            cells = [f"**{c}**" if i > rule and c else c for c in r]
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Reproduce", "", "    python experiments/report_prior_gap.py", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(md))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
