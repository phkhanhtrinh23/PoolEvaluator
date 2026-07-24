"""Compare facility+MMD vs ColBERT+MMD for selecting a subset close to an
unlabeled test set (docs/facility_location_mmd.md, sections 8-10).

We build a CONTROLLED multi-modal, IMBALANCED test distribution with rare modes --
the regime where the theory says the two objectives diverge:

  * facility + MMD  -> submodular COVERAGE (reaches rare modes) + density match
  * ColBERT + MMD   -> modular RELEVANCE (floods the dense mode) + density match
  * ColBERT top-k   -> modular relevance only (the pure retrieval baseline)

Each example is a ColBERT-style MULTI-VECTOR: L unit-norm token vectors. A mode is a
cluster of "topic" token-centroids; rare modes sit far from the dense ones, so an
example's aggregate MaxSim to the (mostly dense) test set is highest for dense-mode
examples -- exactly what makes modular relevance over-pick them.

Run:  python -m experiments.run_subset_match_compare
"""
import numpy as np

from pooleval import subset_match as sm


# --------------------------------------------------------------------------- #
#  Synthetic multi-vector data with imbalanced, multi-modal test distribution #
# --------------------------------------------------------------------------- #
def make_data(seed=0, d=16, L=5, n_modes=5, n_pool=600, n_test=200,
              test_props=(0.55, 0.25, 0.12, 0.05, 0.03)):
    """Return (E_pool, pool_mode, E_test, test_mode, test_props).

    E_* : [n, L, d] unit-norm token vectors.  *_mode : mode label per example.
    Pool is ~uniform over modes (so selection is free to choose any mix); the test
    set is imbalanced per `test_props`, including two rare modes (5% and 3%)."""
    rng = np.random.default_rng(seed)
    test_props = np.asarray(test_props, float)
    test_props = test_props / test_props.sum()

    # Mode "topic" centroids: L token-centroids per mode, spread on the sphere.
    centroids = rng.normal(size=(n_modes, L, d))
    centroids /= np.linalg.norm(centroids, axis=-1, keepdims=True)
    # Push modes apart so rare modes are genuinely distinct from the dense one.
    for k in range(n_modes):
        centroids[k] += 1.6 * rng.normal(size=(1, d))
    centroids /= np.linalg.norm(centroids, axis=-1, keepdims=True)

    def sample(modes):
        E = centroids[modes] + 0.35 * rng.normal(size=(len(modes), L, d))
        E /= np.linalg.norm(E, axis=-1, keepdims=True)
        return E

    pool_mode = rng.integers(0, n_modes, size=n_pool)          # ~uniform pool
    test_mode = rng.choice(n_modes, size=n_test, p=test_props)  # imbalanced test
    return sample(pool_mode), pool_mode, sample(test_mode), test_mode, test_props


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def evaluate(idx, pool_mode, test_props, n_modes, S, K_PP, kt, C):
    idx = np.asarray(idx)
    modes = pool_mode[idx]
    sel_counts = np.bincount(modes, minlength=n_modes).astype(float)
    sel_props = sel_counts / sel_counts.sum()
    rare = np.where(test_props < 0.10)[0]                       # rare test modes
    return {
        "mmd2": sm.mmd2_of(idx, K_PP, kt, C),                  # density match (lower=better)
        "prop_l1": float(np.abs(sel_props - test_props).sum()),# proportion mismatch (lower=better)
        "modes_hit": int((sel_counts > 0).sum()),              # distinct modes covered (of n_modes)
        "rare_hit": int(sum(sel_counts[r] > 0 for r in rare)), # rare modes covered (of len(rare))
        "n_rare": len(rare),
        "redundancy": _intra_maxsim(idx, S, pool_mode),        # dense-mode piling (higher=worse)
        "sel_props": sel_props,
    }


def _intra_maxsim(idx, S, pool_mode):
    """Proxy for redundancy: fraction of the subset landing in its single most
    over-represented mode (1/n_modes = perfectly spread, 1.0 = all one mode)."""
    modes = pool_mode[np.asarray(idx)]
    counts = np.bincount(modes)
    return float(counts.max() / counts.sum())


# --------------------------------------------------------------------------- #
#  Run                                                                         #
# --------------------------------------------------------------------------- #
def main():
    seed = 0
    budget = 40
    beta = 3.0                       # density weight (same for both objectives -> fair)

    E_pool, pool_mode, E_test, test_mode, test_props = make_data(seed=seed)
    n_modes = len(test_props)

    print(f"pool={len(E_pool)}  test={len(E_test)}  modes={n_modes}  budget={budget}  beta={beta}")
    print("test proportions :", np.round(test_props, 3), " (rare modes: <0.10)")

    S = sm.maxsim_matrix(E_test, E_pool)                        # [M,N] MaxSim
    gamma = sm.median_gamma(E_pool, seed=seed)
    K_PP, kt, C = sm.meanmap_blocks(E_pool, E_test, gamma)

    methods = {
        "random":            lambda: list(np.random.default_rng(seed).choice(len(E_pool), budget, replace=False)),
        "ColBERT top-k":     lambda: sm.select_topk_maxsim(S, budget),
        "MMD only":          lambda: sm.select(S, K_PP, kt, C, budget, beta=beta, mode="mmd_only"),
        "ColBERT + MMD":     lambda: sm.select(S, K_PP, kt, C, budget, beta=beta, mode="relevance"),
        "facility + MMD":    lambda: sm.select(S, K_PP, kt, C, budget, beta=beta, mode="facility"),
    }

    rows = {}
    for name, fn in methods.items():
        rows[name] = evaluate(fn(), pool_mode, test_props, n_modes, S, K_PP, kt, C)

    # -- table -------------------------------------------------------------- #
    hdr = f"{'method':<16}{'MMD^2 v':>10}{'prop_L1':>10}{'modes':>8}{'rare':>8}{'dense%':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:<16}{r['mmd2']:>10.4f}{r['prop_l1']:>10.3f}"
              f"{r['modes_hit']:>6}/{n_modes}{r['rare_hit']:>6}/{r['n_rare']}"
              f"{100*r['redundancy']:>8.0f}%")
    print("-" * len(hdr))
    print("MMD^2 v : distribution mismatch to test (lower better)")
    print("prop_L1 : L1 gap of selected vs test mode proportions (lower better)")
    print("modes   : distinct modes covered   rare : rare modes covered")
    print("dense%  : share of subset in its most over-represented mode (lower better)")

    print("\nselected mode proportions (target =", np.round(test_props, 3), "):")
    for name, r in rows.items():
        print(f"  {name:<16}", np.round(r["sel_props"], 3))


if __name__ == "__main__":
    main()
