# Collision-aware EM: detailed derivation

This document derives the collision-aware estimator in
pooleval/new_formulation.py. It assumes no previous knowledge of
expectation-maximization (EM).

## 1. Goal and notation

For target question $i$, the old method chooses one executed table as the fixed
pseudo-label $\hat y_i$. Target gold is withheld while fitting.

- $i=1,\ldots,N$ indexes questions and $j=1,\ldots,J$ indexes models.
- $r_i^j$ is model $j$'s result; $y_i$ is the hidden gold result.
- $C_i^j=1$ if $r_i^j=\hat y_i$; otherwise $C_i^j=0$. We observe $C$.
- $Z_i^j=1$ if $r_i^j=y_i$; otherwise $Z_i^j=0$. We do not observe $Z$.
- $\alpha_j=P(Z_i^j=1)$ is model $j$'s target accuracy.
- $\beta=P(\hat y_i=y_i)$ is pseudo-label accuracy.
- $g(j)$ is model $j$'s provenance group.
- $\gamma_{g(j)}$ is the probability that a wrong model and wrong pseudo-label
  produce the same wrong table.

We learn $\alpha_1,\ldots,\alpha_J$ and $\beta$. We estimate $\gamma_g$ from
labeled source data and hold it fixed on target data.

## 2. The four correctness cases

| Model correct | Pseudo-label correct | Agreement |
| ---: | ---: | --- |
| 1 | 1 | $C=1$: both equal gold |
| 1 | 0 | $C=0$: only the model equals gold |
| 0 | 1 | $C=0$: only the pseudo-label equals gold |
| 0 | 0 | $C$ can be 0 or 1 because wrong tables may differ or collide |

The last row is case 3. In binary classification, two wrong answers must match.
SQL tables are multiclass objects, so we need the collision probability
$\gamma_g$.

## 3. Observation likelihood

When the model is correct, agreement occurs exactly when the pseudo-label is
correct:

$P(C_i^j=1\mid Z_i^j=1)=\beta$.

$P(C_i^j=0\mid Z_i^j=1)=1-\beta$.

When the model is wrong, agreement requires a wrong pseudo-label and a collision:

$P(C_i^j=1\mid Z_i^j=0)=(1-\beta)\gamma_{g(j)}$.

$P(C_i^j=0\mid Z_i^j=0)=1-(1-\beta)\gamma_{g(j)}$.

Define $\gamma_j=\gamma_{g(j)}$ and
$d_j(\beta)=1-(1-\beta)\gamma_j=1-\gamma_j+\beta\gamma_j$. Then:

$P(C_i^j\mid Z_i^j=1)=\beta^{C_i^j}(1-\beta)^{1-C_i^j}$.

$P(C_i^j\mid Z_i^j=0)=[(1-\beta)\gamma_j]^{C_i^j}[d_j(\beta)]^{1-C_i^j}$.

Also, $P(Z_i^j=1\mid\alpha_j)=\alpha_j$ and
$P(Z_i^j=0\mid\alpha_j)=1-\alpha_j$.

## 4. Source-data priors

Let $\pi_j$ be source accuracy and $s_j$ its effective sample size:

$\alpha_j\sim\mathrm{Beta}(1+s_j\pi_j,\ 1+s_j(1-\pi_j))$.

Its parameter-dependent log density is:

$\log p(\alpha_j)=s_j\pi_j\log\alpha_j+s_j(1-\pi_j)\log(1-\alpha_j)$.

Let $\beta_0$ be source pseudo-label accuracy and $s_\beta$ its strength:

$\beta\sim\mathrm{Beta}(1+s_\beta\beta_0,\ 1+s_\beta(1-\beta_0))$.

Its parameter-dependent log density is:

$\log p(\beta)=s_\beta\beta_0\log\beta+s_\beta(1-\beta_0)\log(1-\beta)$.

The added 1 disappears because a Beta density uses powers $a-1$ and $b-1$.
These priors make the result a maximum-a-posteriori estimate.

## 5. Complete-data log posterior

Pretend temporarily that every hidden $Z_i^j$ is known. One cell contributes:

$\ell_{ij}=Z_i^j[\log\alpha_j+C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+(1-Z_i^j)[\log(1-\alpha_j)+C_i^j\log((1-\beta)\gamma_j)+(1-C_i^j)\log d_j(\beta)]$.

After summing all cells and adding the priors:

$\ell_c=\sum_{i,j}Z_i^j[\log\alpha_j+C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+\sum_{i,j}(1-Z_i^j)[\log(1-\alpha_j)+C_i^j\log((1-\beta)\gamma_j)+(1-C_i^j)\log d_j(\beta)]+\sum_j s_j[\pi_j\log\alpha_j+(1-\pi_j)\log(1-\alpha_j)]+s_\beta[\beta_0\log\beta+(1-\beta_0)\log(1-\beta)]$.

Direct maximization is impossible because $Z$ is hidden. EM alternates an
expectation step and a maximization step.

## 6. E-step derivation

At iteration $t$, define:

$\tau_i^j=P(Z_i^j=1\mid C_i^j,\alpha_j^{(t)},\beta^{(t)},\gamma_j)$.

Bayes' rule gives:

$\tau_i^j=\frac{P(Z_i^j=1)P(C_i^j\mid Z_i^j=1)}{P(Z_i^j=1)P(C_i^j\mid Z_i^j=1)+P(Z_i^j=0)P(C_i^j\mid Z_i^j=0)}$.

For $C_i^j=1$:

$\tau_i^j=\frac{\alpha_j\beta}{\alpha_j\beta+(1-\alpha_j)(1-\beta)\gamma_j}$.

For $C_i^j=0$:

$\tau_i^j=\frac{\alpha_j(1-\beta)}{\alpha_j(1-\beta)+(1-\alpha_j)d_j(\beta)}$.

This is the closed-form E-step. Each $\tau_i^j$ is a soft probability that a
model answer is correct.

## 7. Deriving Q

The E-step gives $E[Z_i^j\mid C]=\tau_i^j$. Taking the expectation of
$\ell_c$ replaces $Z_i^j$ by $\tau_i^j$:

$Q(\alpha,\beta)=\sum_{i,j}\tau_i^j[\log\alpha_j+C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+\sum_{i,j}(1-\tau_i^j)[\log(1-\alpha_j)+C_i^j\log((1-\beta)\gamma_j)+(1-C_i^j)\log d_j(\beta)]+\sum_j s_j[\pi_j\log\alpha_j+(1-\pi_j)\log(1-\alpha_j)]+s_\beta[\beta_0\log\beta+(1-\beta_0)\log(1-\beta)]$.

During the M-step, $\tau$ is fixed. We do not differentiate through $\tau$.
The alpha and beta terms separate, so we maximize them independently.

## 8. Alpha M-step: closed-form proof

Keep the terms involving $\alpha_j$:

$Q_{\alpha_j}=\sum_i[\tau_i^j\log\alpha_j+(1-\tau_i^j)\log(1-\alpha_j)]+s_j[\pi_j\log\alpha_j+(1-\pi_j)\log(1-\alpha_j)]$.

Differentiate:

$\frac{\partial Q_{\alpha_j}}{\partial\alpha_j}=\frac{\sum_i\tau_i^j+s_j\pi_j}{\alpha_j}-\frac{N-\sum_i\tau_i^j+s_j(1-\pi_j)}{1-\alpha_j}$.

Set the derivative equal to zero:

$\frac{\sum_i\tau_i^j+s_j\pi_j}{\alpha_j}=\frac{N-\sum_i\tau_i^j+s_j(1-\pi_j)}{1-\alpha_j}$.

Cross-multiply and collect $\alpha_j$:

$(\sum_i\tau_i^j+s_j\pi_j)(1-\alpha_j)=[N-\sum_i\tau_i^j+s_j(1-\pi_j)]\alpha_j$.

$\sum_i\tau_i^j+s_j\pi_j=(N+s_j)\alpha_j$.

Therefore:

$\alpha_j^{(t+1)}=\frac{\sum_i\tau_i^j+s_j\pi_j}{N+s_j}$.

The second derivative is:

$\frac{\partial^2Q_{\alpha_j}}{\partial\alpha_j^2}=-\frac{\sum_i\tau_i^j+s_j\pi_j}{\alpha_j^2}-\frac{N-\sum_i\tau_i^j+s_j(1-\pi_j)}{(1-\alpha_j)^2}\le0$.

Thus this stationary point is a maximum.

## 9. Beta M-step

Keep only the beta terms:

$Q_\beta(\beta)=\sum_{i,j}\tau_i^j[C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+\sum_{i,j}(1-\tau_i^j)[C_i^j\log((1-\beta)\gamma_j)+(1-C_i^j)\log d_j(\beta)]+s_\beta[\beta_0\log\beta+(1-\beta_0)\log(1-\beta)]$.

### 9.1 First derivative

We use:

- $\frac{d}{d\beta}\log\beta=\frac{1}{\beta}$.
- $\frac{d}{d\beta}\log(1-\beta)=-\frac{1}{1-\beta}$.
- $\frac{d}{d\beta}\log((1-\beta)\gamma_j)=-\frac{1}{1-\beta}$.
- $\frac{d}{d\beta}\log d_j(\beta)=\frac{\gamma_j}{d_j(\beta)}$.

Therefore:

$\frac{dQ_\beta}{d\beta}=\sum_{i,j}\tau_i^j[\frac{C_i^j}{\beta}-\frac{1-C_i^j}{1-\beta}]+\sum_{i,j}(1-\tau_i^j)[-\frac{C_i^j}{1-\beta}+\frac{(1-C_i^j)\gamma_j}{d_j(\beta)}]+s_\beta[\frac{\beta_0}{\beta}-\frac{1-\beta_0}{1-\beta}]$.

The maximum satisfies $dQ_\beta/d\beta=0$ inside $0<\beta<1$.

### 9.2 Why beta has no general closed form

Without collision correction, every denominator is $\beta$ or $1-\beta$.
Multiplying by $\beta(1-\beta)$ creates a linear equation and a ratio-of-counts
update.

Collision correction adds:

$\frac{(1-C_i^j)\gamma_j}{1-\gamma_j+\beta\gamma_j}$.

The denominator depends on $\beta$ and the group-specific $\gamma_j$. With
different values of $\gamma_j$, clearing all denominators creates a higher-degree
equation, not one linear count equation. Therefore no general count-ratio closed
form exists.

### 9.3 Concavity proof

Differentiate again:

$\frac{d^2Q_\beta}{d\beta^2}=-\sum_{i,j}\tau_i^j[\frac{C_i^j}{\beta^2}+\frac{1-C_i^j}{(1-\beta)^2}]-\sum_{i,j}(1-\tau_i^j)[\frac{C_i^j}{(1-\beta)^2}+\frac{(1-C_i^j)\gamma_j^2}{d_j(\beta)^2}]-s_\beta[\frac{\beta_0}{\beta^2}+\frac{1-\beta_0}{(1-\beta)^2}]$.

Every bracketed value is non-negative for $0<\beta<1$ and
$0\le\gamma_j\le1$. Hence:

$\frac{d^2Q_\beta}{d\beta^2}\le0$.

Thus $Q_\beta$ is concave. With the source anchor, it is strictly concave and
has one maximum. Numerical optimization cannot become trapped at an inferior
local maximum.

## 10. Numerical beta update in the code

The code searches on:

$10^{-6}\le\beta\le1-10^{-6}$.

Because the scalar routine minimizes, the implementation defines:

$f(\beta)=-Q_\beta(\beta)$.

It minimizes $f$ with SciPy's bounded scalar optimizer. Since $Q_\beta$ is
concave, $-Q_\beta$ is convex. The minimum of $-Q_\beta$ is the global maximum
of $Q_\beta$, up to numerical tolerance.

This is a valid EM M-step. EM requires maximizing $Q$; it does not require a
symbolic closed-form update.

## 11. Worked example

Suppose $C=[1,0]$, $\alpha^{(t)}=0.70$, $\beta^{(t)}=0.80$, and
$\gamma=0.10$.

For the agreement:

$\tau_1=\frac{0.70(0.80)}{0.70(0.80)+0.30(0.20)(0.10)}=\frac{0.56}{0.566}\approx0.9894$.

For the disagreement:

$d(0.80)=1-(1-0.80)(0.10)=0.98$.

$\tau_2=\frac{0.70(0.20)}{0.70(0.20)+0.30(0.98)}=\frac{0.14}{0.434}\approx0.3226$.

If $\pi=0.75$ and $s=10$, then:

$\alpha^{(t+1)}=\frac{0.9894+0.3226+10(0.75)}{2+10}\approx0.7343$.

For beta, freeze $\tau_1$ and $\tau_2$, substitute them into $Q_\beta$, and
maximize the resulting one-dimensional concave function.

## 12. Complete algorithm

1. Select one fixed pseudo-label $\hat y_i$ per target question.
2. Build $C_i^j=1$ when model $j$ agrees with $\hat y_i$.
3. Load source-derived $\pi_j,s_j,\beta_0,s_\beta,\gamma_g$.
4. Initialize $\alpha_j^{(0)}=\pi_j$ and $\beta^{(0)}=\beta_0$.
5. E-step: calculate every $\tau_i^j$ using Bayes' rule above.
6. Alpha M-step: set
   $\alpha_j=(\sum_i\tau_i^j+s_j\pi_j)/(N+s_j)$.
7. Beta M-step: maximize $Q_\beta$ on $[10^{-6},1-10^{-6}]$.
8. Calculate
   $\Delta=\max(\max_j|\alpha_j^{(t+1)}-\alpha_j^{(t)}|,|\beta^{(t+1)}-\beta^{(t)}|)$.
9. Stop when $\Delta<10^{-8}$ or after 200 iterations; otherwise repeat.
10. Recalculate $\tau$ using the returned parameters.

## 13. Summary

- The E-step is closed form because Bayes' rule gives $\tau_i^j$ directly.
- The alpha M-step is closed form because its score equation becomes linear.
- The term $d_j(\beta)=1-\gamma_j+\beta\gamma_j$ removes beta's original
  ratio-of-counts closed form.
- The beta objective is still a concave one-dimensional function, so bounded
  numerical maximization is globally well behaved.
- Numerical maximization is valid EM because the requirement is to maximize
  $Q$, not to express every maximizer as a symbolic fraction.
