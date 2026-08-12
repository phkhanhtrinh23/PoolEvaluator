# Collision-aware EM: detailed derivation

This document derives the collision-aware estimator implemented by
`collision_agreement_em` in `pooleval/new_formulation.py`. It begins from the
probability model and assumes no prior knowledge of expectation-maximization (EM).

## 1. Goal and notation

For every target question (i), the old scoring method uses prior accuracy,
provenance, and the graded kernel verifier to select one executed result table as
the fixed pseudo-label (hat y_i). We observe whether each model agrees with this
pseudo-label, but target gold is withheld while fitting.

- (i=1,ldots,N) indexes questions and (j=1,ldots,J) indexes models.
- (r_i^j) is model (j)'s executed result and (y_i) is the hidden gold result.
- (C_i^j=1) if (r_i^j=hat y_i), otherwise (C_i^j=0). This is observed.
- (Z_i^j=1) if (r_i^j=y_i), otherwise (Z_i^j=0). This is hidden.
- (alpha_j=P(Z_i^j=1)) is model (j)'s target accuracy.
- (eta=P(hat y_i=y_i)) is target pseudo-label accuracy.
- (g(j)) is model (j)'s provenance group.
- (gamma_{g(j)}) is the probability that a wrong model and a wrong
  pseudo-label produce the same wrong table.

The parameters learned on the target set are (alpha_1,ldots,alpha_J) and
(eta). Each (gamma_g) is estimated on labeled source data and then frozen.

## 2. Why case 3 needs (gamma_g)

| Model correct | Pseudo-label correct | Agreement |
| ---: | ---: | --- |
| 1 | 1 | (C=1): both equal gold |
| 1 | 0 | (C=0): only the model equals gold |
| 0 | 1 | (C=0): only the pseudo-label equals gold |
| 0 | 0 | Either value: two wrong tables may match or differ |

The last row is case 3. In a two-class task, two wrong answers must match. SQL
result tables are multiclass objects, so that implication is false. The collision
probability (gamma_g) describes this final row.

## 3. Observation likelihood

If the model is correct, it agrees exactly when the pseudo-label is correct:

(P(C_i^j=1mid Z_i^j=1)=eta),

(P(C_i^j=0mid Z_i^j=1)=1-eta).

If the model is wrong, agreement requires the pseudo-label to be wrong and the two
wrong tables to collide:

(P(C_i^j=1mid Z_i^j=0)=(1-eta)gamma_{g(j)}),

(P(C_i^j=0mid Z_i^j=0)=1-(1-eta)gamma_{g(j)}).

Write (gamma_j=gamma_{g(j)}) and
(d_j(eta)=1-(1-eta)gamma_j=1-gamma_j+etagamma_j). Because (C) is
binary, the likelihoods can be combined as

(P(C_i^jmid Z_i^j=1)=eta^{C_i^j}(1-eta)^{1-C_i^j}),

(P(C_i^jmid Z_i^j=0)=[(1-eta)gamma_j]^{C_i^j}[d_j(eta)]^{1-C_i^j}).

The latent correctness prior is

(P(Z_i^j=1midalpha_j)=alpha_j), and
(P(Z_i^j=0midalpha_j)=1-alpha_j).

## 4. Source-data anchors

Target binary agreements alone have a mirror ambiguity, so labeled source
information remains in the objective rather than serving only as initialization.
Let (pi_j) be source accuracy and (s_j) its effective sample size:

(alpha_jsimmathrm{Beta}(1+s_jpi_j, 1+s_j(1-pi_j))).

Ignoring constants, its log density is

(s_j[pi_jlogalpha_j+(1-pi_j)log(1-alpha_j)]).

Likewise, source pseudo-label accuracy (eta_0) and strength (s_eta) give

(etasimmathrm{Beta}(1+s_etaeta_0, 1+s_eta(1-eta_0))),

with log-density contribution

(s_eta[eta_0logeta+(1-eta_0)log(1-eta)]).

The added 1 disappears because a Beta density uses exponents (a-1) and (b-1).
These priors make the estimate maximum-a-posteriori, abbreviated MAP.

## 5. Complete-data log posterior

Suppose temporarily that every hidden (Z_i^j) were known. One cell contributes

(Z_i^j[logalpha_j+C_i^jlogeta+(1-C_i^j)log(1-eta)])

(+(1-Z_i^j)[log(1-alpha_j)+C_i^jlog((1-eta)gamma_j)+(1-C_i^j)log d_j(eta)]).

After summing all cells and adding source priors, the complete-data log posterior,
up to constants independent of the learned parameters, is

(ell_c=sum_{i,j}Z_i^j[logalpha_j+C_i^jlogeta+(1-C_i^j)log(1-eta)])

(+sum_{i,j}(1-Z_i^j)[log(1-alpha_j)+C_i^jlog((1-eta)gamma_j)+(1-C_i^j)log d_j(eta)])

(+sum_j s_j[pi_jlogalpha_j+(1-pi_j)log(1-alpha_j)])

(+s_eta[eta_0logeta+(1-eta_0)log(1-eta)]).

Direct maximization is impossible because (Z) is hidden. EM handles this by
alternating an expectation step and a maximization step.

## 6. E-step proof

At iteration (t), define the posterior probability

(	au_i^j=P(Z_i^j=1mid C_i^j,alpha_j^{(t)},eta^{(t)},gamma_j)).

Bayes' rule gives

(	au_i^j=rac{P(Z_i^j=1)P(C_i^jmid Z_i^j=1)}{P(Z_i^j=1)P(C_i^jmid Z_i^j=1)+P(Z_i^j=0)P(C_i^jmid Z_i^j=0)}).

For an agreement, (C_i^j=1), substitution produces

(	au_i^j=rac{alpha_jeta}{alpha_jeta+(1-alpha_j)(1-eta)gamma_j}).

For a disagreement, (C_i^j=0), it produces

(	au_i^j=rac{alpha_j(1-eta)}{alpha_j(1-eta)+(1-alpha_j)d_j(eta)}).

Thus the E-step is closed form. It assigns soft correctness probabilities rather
than forcing every model answer to be correct or wrong.

## 7. Deriving the auxiliary objective (Q)

The E-step implies (E[Z_i^jmid C]=	au_i^j) and
(E[1-Z_i^jmid C]=1-	au_i^j). Taking the expectation of (ell_c) simply
replaces (Z_i^j) by (	au_i^j):

(Q(alpha,eta)=sum_{i,j}	au_i^j[logalpha_j+C_i^jlogeta+(1-C_i^j)log(1-eta)])

(+sum_{i,j}(1-	au_i^j)[log(1-alpha_j)+C_i^jlog((1-eta)gamma_j)+(1-C_i^j)log d_j(eta)])

(+sum_j s_j[pi_jlogalpha_j+(1-pi_j)log(1-alpha_j)])

(+s_eta[eta_0logeta+(1-eta_0)log(1-eta)]).

In the M-step, (	au) is fixed at its E-step value. We must not differentiate
through (	au). The (alpha) and (eta) terms separate, so we maximize them
independently.

## 8. Closed-form M-step for (alpha_j)

The terms containing one (alpha_j) are

(Q_{alpha_j}=sum_i[	au_i^jlogalpha_j+(1-	au_i^j)log(1-alpha_j)]+s_j[pi_jlogalpha_j+(1-pi_j)log(1-alpha_j)]).

Its derivative is

(rac{partial Q_{alpha_j}}{partialalpha_j}=rac{sum_i	au_i^j+s_jpi_j}{alpha_j}-rac{N-sum_i	au_i^j+s_j(1-pi_j)}{1-alpha_j}).

Set it to zero and multiply by (alpha_j(1-alpha_j)):

((sum_i	au_i^j+s_jpi_j)(1-alpha_j)=[N-sum_i	au_i^j+s_j(1-pi_j)]alpha_j).

Expanding and collecting terms yields

(sum_i	au_i^j+s_jpi_j=(N+s_j)alpha_j),

so the exact update is

(alpha_j^{(t+1)}=rac{sum_i	au_i^j+s_jpi_j}{N+s_j}).

Its second derivative is

(rac{partial^2Q_{alpha_j}}{partialalpha_j^2}=-rac{sum_i	au_i^j+s_jpi_j}{alpha_j^2}-rac{N-sum_i	au_i^j+s_j(1-pi_j)}{(1-alpha_j)^2}le0),

so this stationary point is a maximum. When (s_j=0), the update reduces to the
mean posterior correctness (N^{-1}sum_i	au_i^j).

## 9. M-step for (eta)

The exact part of (Q) that depends on (eta) is

(Q_eta(eta)=sum_{i,j}	au_i^j[C_i^jlogeta+(1-C_i^j)log(1-eta)])

(+sum_{i,j}(1-	au_i^j)[C_i^jlog((1-eta)gamma_j)+(1-C_i^j)log d_j(eta)])

(+s_eta[eta_0logeta+(1-eta_0)log(1-eta)]).

### 9.1 First derivative

The needed elementary derivatives are

- (dlogeta/deta=1/eta);
- (dlog(1-eta)/deta=-1/(1-eta));
- (dlog((1-eta)gamma_j)/deta=-1/(1-eta));
- (dlog d_j(eta)/deta=gamma_j/d_j(eta)).

Therefore

(rac{dQ_eta}{deta}=sum_{i,j}	au_i^j[rac{C_i^j}{eta}-rac{1-C_i^j}{1-eta}])

(+sum_{i,j}(1-	au_i^j)[-rac{C_i^j}{1-eta}+rac{(1-C_i^j)gamma_j}{d_j(eta)}])

(+s_eta[rac{eta_0}{eta}-rac{1-eta_0}{1-eta}]).

The maximizing (eta) is the root of this score equation within (0<eta<1).

### 9.2 Why there is no general count-ratio closed form

Without the collision correction, every denominator is either (eta) or
(1-eta). Multiplying the score equation by (eta(1-eta)) gives a linear
equation, hence the original ratio-of-expected-counts update.

The corrected derivative additionally contains

(rac{(1-C_i^j)gamma_j}{1-gamma_j+etagamma_j}).

Its denominator depends on both (eta) and the group-specific (gamma_j). With
several different (gamma_j), clearing every denominator creates a higher-degree
equation, not one linear count equation. Consequently there is no general symbolic
ratio update. A numerical root or numerical maximization is the correct M-step.

The special case (gamma_j=1) for all models gives (d_j(eta)=eta) and
recovers the original binary symmetry, but real source-estimated collision rates
are not all one.

### 9.3 Concavity proof: why numerical optimization is reliable

Differentiate again:

(rac{d^2Q_eta}{deta^2}=-sum_{i,j}	au_i^j[rac{C_i^j}{eta^2}+rac{1-C_i^j}{(1-eta)^2}])

(-sum_{i,j}(1-	au_i^j)[rac{C_i^j}{(1-eta)^2}+rac{(1-C_i^j)gamma_j^2}{d_j(eta)^2}])

(-s_eta[rac{eta_0}{eta^2}+rac{1-eta_0}{(1-eta)^2}]).

Every bracketed quantity is non-negative for (0<eta<1) and
(0legamma_jle1). Hence (d^2Q_eta/deta^2le0), so (Q_eta) is
concave. Except in degenerate cases it is strictly concave and has only one
maximum. There is no inferior local maximum in which a bounded one-dimensional
optimizer could become trapped.

## 10. Exact numerical procedure used by the code

Logarithms are undefined at probabilities zero and one. The implementation searches

(10^{-6}leetale1-10^{-6}).

SciPy's bounded scalar routine minimizes, so the code defines

(f(eta)=-Q_eta(eta))

and uses `minimize_scalar(f, bounds=(eps, 1-eps), method="bounded")`. Since
(Q_eta) is concave, (-Q_eta) is convex. Minimizing this scalar function is
equivalent to globally maximizing (Q_eta), to numerical tolerance.

This remains a genuine EM M-step. EM requires maximization of (Q); it does not
require a symbolic closed-form update.

## 11. Worked E-step and alpha M-step

Consider one model and two items with
(C=[1,0]), (alpha^{(t)}=0.70), (eta^{(t)}=0.80), and (gamma=0.10).

For the agreement cell,

(	au_1=rac{0.70(0.80)}{0.70(0.80)+0.30(0.20)(0.10)}=rac{0.56}{0.566}approx0.9894).

For the disagreement cell,

(d(0.80)=1-(1-0.80)(0.10)=0.98),

(	au_2=rac{0.70(0.20)}{0.70(0.20)+0.30(0.98)}=rac{0.14}{0.434}approx0.3226).

If (pi=0.75) and (s=10), then

(alpha^{(t+1)}=rac{0.9894+0.3226+10(0.75)}{2+10}approx0.7343).

For the beta update, freeze (	au_1,	au_2), substitute them and the source beta
prior into (Q_eta), and maximize that one-dimensional concave function. The
next E-step then uses both newly updated parameters.

## 12. Complete algorithm

1. Select one fixed pseudo-label per target item using the old scoring method.
2. Build the binary matrix (C_i^j=1) when model (j)'s executed table equals
   that pseudo-label.
3. Load source-derived (pi_j,s_j,eta_0,s_eta,gamma_g).
4. Initialize (alpha_j^{(0)}=pi_j) and (eta^{(0)}=eta_0).
5. E-step: compute each (	au_i^j) using Bayes' rule above.
6. M-step: update every
   (alpha_j=(sum_i	au_i^j+s_jpi_j)/(N+s_j)).
7. M-step: maximize (Q_eta) numerically on
   ([10^{-6},1-10^{-6}]).
8. Compute
   (Delta=max(max_j|alpha_j^{(t+1)}-alpha_j^{(t)}|, |eta^{(t+1)}-eta^{(t)}|)).
9. Stop if (Delta<10^{-8}) or after 200 iterations; otherwise repeat step 5.
10. Recompute (	au) once at the returned parameters so all outputs describe the
    same final state.

## 13. What is observed, fixed, and learned?

| Quantity | Role | Source |
| --- | --- | --- |
| (C_i^j) | Observed agreement | Target executions without gold |
| (pi_j,s_j) | Model accuracy anchor | Labeled source/meta data |
| (eta_0,s_eta) | Pseudo-label anchor | Labeled source/meta data |
| (gamma_g) | Fixed wrong-table collision rate | Labeled source/meta data |
| (	au_i^j) | Soft correctness | Recomputed in every E-step |
| (alpha_j) | Target model accuracy | Learned in the M-step |
| (eta) | Target pseudo-label accuracy | Learned by scalar M-step |

Target gold is used only afterward to score the experiment. It is not used to
construct (C), calculate (	au), or update (alpha) or (eta).

## 14. Summary

- The E-step is closed form because Bayes' rule directly gives (	au_i^j).
- The (alpha_j) M-step is closed form because its score equation becomes linear.
- The term (d_j(eta)=1-gamma_j+etagamma_j) removes the original closed-form
  count ratio for (eta).
- The beta objective remains a concave one-dimensional function. Bounded numerical
  maximization is therefore inexpensive and globally well behaved.
- Numerical maximization is fully valid EM: the requirement is maximizing (Q),
  not writing every maximizer as a symbolic fraction.
