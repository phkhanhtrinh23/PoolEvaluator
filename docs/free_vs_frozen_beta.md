# "Free EM-fitted $\beta$" vs "frozen measured $\hat\beta$" — what that actually means

You asked what I meant by that sentence. Here it is with real numbers from spider, no symbols
left undefined.

## The one number both arms are trying to compute

Both arms need the same quantity for the E-step:

$$
P\big(C_i^j = 1 \mid Z_i^j = 0\big)
\;=\;
\text{"given model } j \text{ is WRONG, how often does it still AGREE with the pseudo-label?"}
$$

Both build it the same way, as a product of two factors:

$$
P(C=1\mid Z=0)
\;=\;
\underbrace{P(\hat y \text{ is also wrong})}_{\text{factor 1}}
\;\times\;
\underbrace{P(\text{they give the SAME wrong answer})}_{\text{factor 2}}
$$

**Factor 2 is identical in both arms.** It is the collision statistic, measured by counting on
the labeled split:

```
gamma_coll (models 0..4) : [0.952 0.913 0.783 0.667 0.955]
gamma_both (models 0..4) : [0.048 0.087 0.217 0.333 0.045]     (= 1 - gamma_coll)
```

Same counts, complementary direction. Nothing separates the two arms here.

**Factor 1 is where they differ.** It is "how often is the pseudo-label wrong", i.e.
$1-\beta$. And the two arms get that number from completely different places.

---

## `both_wrong`: the number is COUNTED, then held still

$\hat\beta$ is measured once, on the labeled split, by counting how often the vote's winner
equals gold:

```
beta_hat = 0.8115        <- a plain count. One scalar. No EM involved.
```

During EM this number **does not move**. Sweep 1, sweep 50, sweep 200 — the E-step sees the
same 0.8115 every time. So factor 1 is fixed at $1-0.8115 = 0.1885$, and

```
both_wrong, at EVERY sweep:
  P(C=1|Z=0) for models 0..4 = [0.1795 0.1721 0.1475 0.1257 0.1800]
```

That is what **frozen** means: an input to EM, not an output of it. It is computed *before* EM
starts and EM has no ability to change it.

*(It is not frozen forever — the expert updates it by $(\text{old}+\text{temp})/2$ after each
validated item. Frozen means "constant across the EM sweeps", not "constant for all time".)*

---

## `collision`: the number is FITTED, and EM moves it

Here $\beta$ is a **free parameter**. EM is allowed to choose whatever value maximises the
likelihood, and re-chooses it in every single M-step. Watch what it does:

| after … EM sweeps | fitted $\beta$ | resulting $P(C=1\mid Z=0)$, models 0–4 |
|---|---:|---|
| start | 0.7000 | — |
| 1 | 0.8628 | 0.1307  0.1253  0.1074  0.0915  0.1310 |
| 3 | 0.9660 | 0.0324  0.0311  0.0266  0.0227  0.0325 |
| 10 | 0.9988 | 0.0011  0.0011  0.0009  0.0008  0.0011 |
| 200 | **1.0000** | **0.0000  0.0000  0.0000  0.0000  0.0000** |

That is what **free / EM-fitted** means: an output of EM, re-estimated every sweep, and here it
walks from 0.7 all the way to the boundary at 1.0.

And when $\beta$ reaches 1, factor 1 is $1-\beta = 0$. **Zero times anything is zero.** The
carefully measured $\gamma^{\text{coll}}$ — 0.952, 0.913, 0.783, … — gets multiplied by 0 and
contributes nothing. The model now asserts *"a wrong classifier never agrees with the
pseudo-label"*, so agreement becomes a perfect certificate of correctness.

---

## The decisive comparison

Here is the part that settles it. **`both_wrong` fits its own $\beta$ too, and that $\beta$
also goes to 1.0000.**

```
both_wrong's OWN fitted beta after 200 sweeps: 1.0000
```

Identical degeneracy, same dataset, same sweeps. But in `both_wrong` that fitted $\beta$ lives
**only in the $Z=1$ branch** — it describes how often a *correct* model agrees with the
pseudo-label. It never touches $\gamma$. The $Z=0$ branch is built from the frozen
$\hat\beta = 0.8115$ and stays at 0.1795 regardless.

So the two arms have the *same* pathological $\beta$. The difference is purely **where that
$\beta$ is allowed to act**:

| | `collision` | `both_wrong` |
|---|---|---|
| statistic measured | $\gamma^{\text{coll}}$ | $\gamma^{\text{both}}$ (same counts) |
| $Z=1$ branch uses | fitted $\beta \to 1.0$ | fitted $\beta \to 1.0$ |
| $Z=0$ branch uses | **the same fitted $\beta$** | **frozen $\hat\beta = 0.8115$** |
| $P(C=1\mid Z=0)$ ends at | **0.0000** | 0.1795 |
| can the expert's $\gamma$ update reach the likelihood? | no, scaled by $1-\beta=0$ | yes, scaled by $1-\hat\beta=0.1885$ |
| MAE at budget 40 (3-modality mean) | 17.11 | **6.67** |

---

## Restating the original sentence

> "collision uses the free EM-fitted $\beta$, both_wrong uses the frozen measured $\hat\beta$"

means: **both arms need to know how often the pseudo-label is wrong, and collision asks EM to
guess it while both_wrong looks it up from counted data.** EM's guess is unidentified — the
likelihood is nearly flat in that direction — so the guess runs to the boundary, and at the
boundary it annihilates the collision statistic it was supposed to be multiplying.

The frozen $\hat\beta$ is worse in one respect: it is measured on the labeled split, so under
domain shift it is somewhat wrong. But "somewhat wrong and non-zero" is enormously better than
"optimally fitted and exactly zero", because only the second one deletes $\gamma$ from the
model.
