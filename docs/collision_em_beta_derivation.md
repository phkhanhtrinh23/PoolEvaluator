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

## 7. What is Q, and why does EM need it?

### 7.1 The function we really want to maximize

Write all unknown parameters as $\theta=(\alpha_1,\ldots,\alpha_J,\beta)$.
Ideally, we would maximize the observed-data log posterior:

$L(\theta)=\log P(C\mid\theta)+\log p(\theta)$.

The problem is that $P(C\mid\theta)$ must sum over every possible value of the
hidden matrix $Z$:

$P(C\mid\theta)=\sum_Z P(C,Z\mid\theta)$.

Therefore:

$L(\theta)=\log\sum_Z P(C,Z\mid\theta)+\log p(\theta)$.

The logarithm is outside the sum. This makes direct differentiation difficult:
every possible hidden explanation $Z$ is mixed inside one logarithm.

### 7.2 EM creates a temporary distribution over Z

At iteration $t$, the E-step calculates the posterior distribution of the hidden
variables using the current parameters $\theta^{(t)}$:

$q_t(Z)=P(Z\mid C,\theta^{(t)})$.

For one cell, this distribution is summarized by:

$q_t(Z_i^j=1)=\tau_i^j$.

$q_t(Z_i^j=0)=1-\tau_i^j$.

The letter $q$ here means a temporary probability distribution over possible
hidden correctness values. It is not the same thing as the auxiliary function
$Q$.

### 7.3 The lower-bound reason Q exists

Insert $q_t(Z)/q_t(Z)=1$ inside the observed likelihood:

$\log P(C\mid\theta)=\log\sum_Z q_t(Z)\frac{P(C,Z\mid\theta)}{q_t(Z)}$.

Because the logarithm is concave, Jensen's inequality gives:

$\log P(C\mid\theta)\ge\sum_Z q_t(Z)\log\frac{P(C,Z\mid\theta)}{q_t(Z)}$.

After adding the log prior, the lower bound is:

$L(\theta)\ge\sum_Z q_t(Z)\log P(C,Z\mid\theta)+\log p(\theta)-\sum_Z q_t(Z)\log q_t(Z)$.

Define the first two parameter-dependent terms as:

$Q(\theta\mid\theta^{(t)})=E_{Z\sim q_t}[\log P(C,Z\mid\theta)]+\log p(\theta)$.

Define the remaining term as the entropy:

$H(q_t)=-\sum_Z q_t(Z)\log q_t(Z)$.

Then:

$L(\theta)\ge Q(\theta\mid\theta^{(t)})+H(q_t)$.

During the M-step, $q_t$ and therefore $H(q_t)$ are fixed. Maximizing the lower
bound with respect to $\theta$ is consequently the same as maximizing $Q$.
The E-step chooses the exact posterior $q_t$, which makes the bound touch the
true objective at the current parameters. The M-step raises this touching lower
bound. This is why ordinary EM does not decrease the observed-data objective.

In plain language, $Q$ asks:

> If the current soft beliefs about which model answers are correct were true,
> which new values of alpha and beta would make those explanations most likely?

### 7.4 Why Q has this particular formula

The complete-data log posterior $\ell_c$ in section 5 contains $Z_i^j$ whenever
the cell is explained as a correct-model case and $1-Z_i^j$ whenever it is
explained as a wrong-model case. The E-step gives:

$E[Z_i^j\mid C,\theta^{(t)}]=\tau_i^j$.

$E[1-Z_i^j\mid C,\theta^{(t)}]=1-\tau_i^j$.

Taking the expectation of $\ell_c$ therefore replaces each $Z_i^j$ by
$\tau_i^j$ and each $1-Z_i^j$ by $1-\tau_i^j$:

$Q(\alpha,\beta)=\sum_{i,j}\tau_i^j[\log\alpha_j+C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+\sum_{i,j}(1-\tau_i^j)[\log(1-\alpha_j)+C_i^j\log((1-\beta)\gamma_j)+(1-C_i^j)\log d_j(\beta)]+\sum_j s_j[\pi_j\log\alpha_j+(1-\pi_j)\log(1-\alpha_j)]+s_\beta[\beta_0\log\beta+(1-\beta_0)\log(1-\beta)]$.

Every part has a direct source:

| Part of $Q$ | Why it appears |
| --- | --- |
| $\tau_i^j\log\alpha_j$ | Soft weight for the prior probability that model $j$ is correct |
| $(1-\tau_i^j)\log(1-\alpha_j)$ | Soft weight for the probability that model $j$ is wrong |
| $\tau_i^j C_i^j\log\beta$ | Correct model agrees, so the pseudo-label must be correct |
| $\tau_i^j(1-C_i^j)\log(1-\beta)$ | Correct model disagrees, so the pseudo-label must be wrong |
| $(1-\tau_i^j)C_i^j\log((1-\beta)\gamma_j)$ | Wrong model agrees only through a wrong pseudo-label and collision |
| $(1-\tau_i^j)(1-C_i^j)\log d_j(\beta)$ | Wrong model does not collide with the pseudo-label |
| Terms multiplied by $s_j$ | Source-data prior for each $\alpha_j$ |
| Terms multiplied by $s_\beta$ | Source-data prior for $\beta$ |

During the M-step, $\tau$ is fixed at its E-step value. We do not differentiate
through $\tau$. The alpha and beta terms separate, so we maximize them
independently.

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

### 9.2 Substitute the known gamma values

The experiment knows every $\gamma_g$ before target EM begins. To make the
resulting beta equation explicit, define three sets of known expected counts from
the current E-step:

$A=\sum_{i,j}\tau_i^jC_i^j+s_\beta\beta_0$.

$B=\sum_{i,j}[\tau_i^j(1-C_i^j)+(1-\tau_i^j)C_i^j]+s_\beta(1-\beta_0)$.

$D_g=\sum_{i,j:g(j)=g}(1-\tau_i^j)(1-C_i^j)$.

Here, $A$ collects terms whose derivative contributes $1/\beta$, $B$ collects
terms whose derivative contributes $-1/(1-\beta)$, and $D_g$ collects
wrong-model disagreements for group $g$. During this M-step, $\tau$, $C$,
$s_\beta$, $\beta_0$, and every $\gamma_g$ are known constants.

Using these definitions, the full score equation becomes:

$\frac{dQ_\beta}{d\beta}=\frac{A}{\beta}-\frac{B}{1-\beta}+\sum_g\frac{D_g\gamma_g}{1-\gamma_g+\beta\gamma_g}=0$.

This equation is the exact solution condition for beta after the known gamma
values have been substituted.

### 9.3 General polynomial whose unique root is beta

Define $d_g(\beta)=1-\gamma_g+\beta\gamma_g$. Multiply the score equation by
$\beta(1-\beta)\prod_gd_g(\beta)$. The denominators disappear and give:

$P(\beta)=A(1-\beta)\prod_gd_g(\beta)-B\beta\prod_gd_g(\beta)+\beta(1-\beta)\sum_g[D_g\gamma_g\prod_{h\ne g}d_h(\beta)]=0$.

All coefficients of $P$ are known during the M-step. Therefore the beta update
can be stated exactly as:

$\beta^{(t+1)}=\text{the unique root of }P(\beta)=0\text{ in }(0,1)$.

If there are $G$ distinct gamma values, $P$ generally has degree $G+1$.
Knowing the coefficients makes its root numerically computable, but it does not
create a universal symbolic formula for arbitrary $G$. In particular, general
polynomials of degree five or higher do not have a formula using only arithmetic
and radicals. This is why knowing gamma does not by itself restore the original
count-ratio update.

### 9.4 Closed form when all groups share one gamma

If every model uses one common known collision probability $\gamma$, define
$D=\sum_gD_g$. The score equation is:

$\frac{A}{\beta}-\frac{B}{1-\beta}+\frac{D\gamma}{1-\gamma+\beta\gamma}=0$.

After multiplying by $\beta(1-\beta)(1-\gamma+\beta\gamma)$, we obtain:

$c_2\beta^2+c_1\beta+c_0=0$.

The known coefficients are:

$c_2=-\gamma(A+B+D)$.

$c_1=A(2\gamma-1)-B(1-\gamma)+D\gamma$.

$c_0=A(1-\gamma)$.

For $\gamma>0$, the two algebraic candidates are:

$\beta_\pm=\frac{-c_1\pm\sqrt{c_1^2-4c_2c_0}}{2c_2}$.

The concavity proof below guarantees that exactly one valid maximizing candidate
lies in $(0,1)$; that candidate is $\beta^{(t+1)}$. If $\gamma=0$, the collision
term vanishes and the solution simplifies to:

$\beta^{(t+1)}=\frac{A}{A+B}$.

Thus a genuine quadratic closed form exists for one shared gamma. The implemented
experiment uses provenance-specific $\gamma_g$, so it solves the general unique
root rather than incorrectly replacing all group collision rates by one number.

### 9.5 Concavity proof

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

## 14. Target-accuracy guarantee from a distribution-aligned prior

This section establishes a bound connecting the accuracy measured on the selected labeled source subsets to the unknown accuracy of the same model on the unlabeled target workload.

The source subsets are selected to resemble the target workload using two complementary criteria:

1. facility-location coverage, which encourages the selected source items to cover the important regions of the target workload; and
2. MMD, which explicitly measures the distributional discrepancy between the selected source items and the target items in an RKHS.

The main result shows that, under an RKHS regularity assumption, a small MMD implies that the model accuracy on the selected source data cannot differ arbitrarily from its accuracy on the target data.

### 14.1 Setup

Let $\mathcal S$ denote the labeled source collection, such as MetaDataset.

Let

$X=\{x_1,\ldots,x_T\}$

denote the unlabeled target workload.

The source collection is partitioned into candidate subsets

$\mathcal P=\{P_1,\ldots,P_M\}.$

We select at most $K$ source subsets by balancing target coverage and distribution matching:

$A^\star=\arg\max_{A\subseteq\mathcal P,\ |A|\le K}\big[f_{\mathrm{cov}}(A)-\lambda_{\mathrm{MMD}}\mathrm{MMD}^2(U_A,X)\big].$

Here,

$U_A=\bigcup_{P\in A}P.$

Let the finally selected source set be

$U=U_{A^\star}.$

Suppose $U$ contains $n$ labeled source items:

$U=\{z_1,\ldots,z_n\}.$

The facility-location term $f_{\mathrm{cov}}(A)$ encourages the selected source set to cover the target workload, including target regions that may contain relatively few items.

The MMD term encourages the distribution represented by $U$ to match the distribution represented by $X$.

We now define empirical source and target distributions.

Let $P_S$ assign probability $1/n$ to every selected source item $z_r$.

Let $P_T$ assign probability $1/T$ to every target item $x_i$.

Therefore, $P_S$ and $P_T$ are the empirical distributions of the selected source set and target workload, respectively.

For model $F_j$, define its correctness function $f_j(a)$.

For deterministic Text-to-SQL evaluation,

$f_j(a)=1$

when model $F_j$ produces the correct execution result on item $a$, and

$f_j(a)=0$

otherwise.

Thus,

$0\le f_j(a)\le1.$

Because the selected source items are labeled, their correctness values are observable.

The prior accuracy calculated from the selected source set is

$\hat \pi_j=\frac{1}{n}\sum_{r=1}^{n}f_j(z_r).$

Since $P_S$ is the empirical distribution over the selected source items, this can equivalently be written as

$\hat \pi_j=E_{a\sim P_S}[f_j(a)].$

The target labels are unavailable, so the target accuracy is unknown.

Define the true accuracy of model $F_j$ on the fixed target workload as

$\alpha_j^\star=\frac{1}{T}\sum_{i=1}^{T}f_j(x_i).$

Since $P_T$ is the empirical distribution over the target workload,

$\alpha_j^\star=E_{a\sim P_T}[f_j(a)].$

Our objective is to use the observed prior accuracy $\hat \pi_j$ and the source-target MMD to bound the unknown target accuracy $\alpha_j^\star$.

### 14.2 RKHS assumption

Let $k$ denote the positive-semidefinite RBF kernel already used to compute MMD.

Let $\mathcal H$ denote the RKHS induced by $k$.

Let $\psi(a)$ denote the corresponding feature representation satisfying

$k(a,b)=\langle\psi(a),\psi(b)\rangle_{\mathcal H}.$

We make the following regularity assumption.

**Assumption 1.**

The correctness function $f_j$ belongs to the RKHS $\mathcal H$:

$f_j\in\mathcal H.$

Furthermore, there exists a finite constant $B_j$ satisfying

$\sqrt{\langle f_j,f_j\rangle_{\mathcal H}}\le B_j.$

This assumption is important.

MMD measures whether two distributions are close with respect to functions represented by the chosen RKHS.

Therefore, to use MMD to control the change in model accuracy, the model correctness function must be sufficiently regular under the same RKHS.

Without this assumption, two distributions may have small MMD while the model behaves very differently on them.

### 14.3 Kernel mean embeddings

Define the empirical kernel mean embedding of the selected source distribution as

$\mu_S=E_{a\sim P_S}[\psi(a)].$

Since $P_S$ is uniform over the selected source items,

$\mu_S=\frac{1}{n}\sum_{r=1}^{n}\psi(z_r).$

Similarly, define the empirical kernel mean embedding of the target distribution as

$\mu_T=E_{a\sim P_T}[\psi(a)].$

Since $P_T$ is uniform over the target workload,

$\mu_T=\frac{1}{T}\sum_{i=1}^{T}\psi(x_i).$

The MMD between the selected source set and target workload is

$\mathrm{MMD}(P_S,P_T)=\sqrt{\langle\mu_S-\mu_T,\mu_S-\mu_T\rangle_{\mathcal H}}.$

This is exactly the RKHS distance between the source and target mean embeddings.

### 14.4 Expressing source accuracy in the RKHS

Because $f_j\in\mathcal H$, the reproducing property gives

$f_j(a)=\langle f_j,\psi(a)\rangle_{\mathcal H}.$

Recall that the observed source accuracy is

$\hat \pi_j=E_{a\sim P_S}[f_j(a)].$

Substituting the reproducing property gives

$\hat \pi_j=E_{a\sim P_S}[\langle f_j,\psi(a)\rangle_{\mathcal H}].$

Using linearity of expectation,

$\hat \pi_j=\langle f_j,E_{a\sim P_S}[\psi(a)]\rangle_{\mathcal H}.$

Since

$E_{a\sim P_S}[\psi(a)]=\mu_S,$

we obtain

$\hat \pi_j=\langle f_j,\mu_S\rangle_{\mathcal H}.$

### 14.5 Expressing target accuracy in the RKHS

The unknown target accuracy is

$\alpha_j^\star=E_{a\sim P_T}[f_j(a)].$

Again using the reproducing property,

$\alpha_j^\star=E_{a\sim P_T}[\langle f_j,\psi(a)\rangle_{\mathcal H}].$

Using linearity of expectation,

$\alpha_j^\star=\langle f_j,E_{a\sim P_T}[\psi(a)]\rangle_{\mathcal H}.$

Since

$E_{a\sim P_T}[\psi(a)]=\mu_T,$

we obtain

$\alpha_j^\star=\langle f_j,\mu_T\rangle_{\mathcal H}.$

### 14.6 Deriving the upper target-accuracy bound

Subtracting the source accuracy from the target accuracy gives

$\alpha_j^\star-\hat \pi_j=\langle f_j,\mu_T-\mu_S\rangle_{\mathcal H}.$

By the Cauchy-Schwarz inequality,

$\langle f_j,\mu_T-\mu_S\rangle_{\mathcal H}\le\sqrt{\langle f_j,f_j\rangle_{\mathcal H}}\sqrt{\langle\mu_T-\mu_S,\mu_T-\mu_S\rangle_{\mathcal H}}.$

From Assumption 1,

$\sqrt{\langle f_j,f_j\rangle_{\mathcal H}}\le B_j.$

Therefore,

$\alpha_j^\star-\hat \pi_j\le B_j\sqrt{\langle\mu_T-\mu_S,\mu_T-\mu_S\rangle_{\mathcal H}}.$

By the definition of MMD,

$\sqrt{\langle\mu_T-\mu_S,\mu_T-\mu_S\rangle_{\mathcal H}}=\mathrm{MMD}(P_S,P_T).$

Hence,

$\alpha_j^\star-\hat \pi_j\le B_j\mathrm{MMD}(P_S,P_T).$

Rearranging gives

$\alpha_j^\star\le\hat \pi_j+B_j\mathrm{MMD}(P_S,P_T).$

Therefore, the target accuracy cannot exceed the source prior accuracy by more than the MMD penalty under the stated RKHS assumption.

### 14.7 Deriving the lower target-accuracy bound

We now reverse the difference.

Subtracting target accuracy from source accuracy gives

$\hat \pi_j-\alpha_j^\star=\langle f_j,\mu_S-\mu_T\rangle_{\mathcal H}.$

By the Cauchy-Schwarz inequality,

$\langle f_j,\mu_S-\mu_T\rangle_{\mathcal H}\le\sqrt{\langle f_j,f_j\rangle_{\mathcal H}}\sqrt{\langle\mu_S-\mu_T,\mu_S-\mu_T\rangle_{\mathcal H}}.$

Using Assumption 1,

$\hat \pi_j-\alpha_j^\star\le B_j\sqrt{\langle\mu_S-\mu_T,\mu_S-\mu_T\rangle_{\mathcal H}}.$

The RKHS distance does not depend on the order of subtraction, so

$\sqrt{\langle\mu_S-\mu_T,\mu_S-\mu_T\rangle_{\mathcal H}}=\mathrm{MMD}(P_S,P_T).$

Therefore,

$\hat \pi_j-\alpha_j^\star\le B_j\mathrm{MMD}(P_S,P_T).$

Rearranging gives

$\alpha_j^\star\ge\hat \pi_j-B_j\mathrm{MMD}(P_S,P_T).$

This is the desired lower target-accuracy bound.

### 14.8 Main deterministic theorem

**Theorem 1. Target-accuracy bound from a distribution-aligned source prior.**

Suppose the correctness function of model $F_j$ satisfies Assumption 1.

Let $\hat \pi_j$ denote its observed execution accuracy on the selected labeled source set.

Let $\alpha_j^\star$ denote its unknown execution accuracy on the fixed unlabeled target workload.

Then

$\alpha_j^\star\ge\hat \pi_j-B_j\mathrm{MMD}(P_S,P_T)$

and

$\alpha_j^\star\le\hat \pi_j+B_j\mathrm{MMD}(P_S,P_T).$

Therefore,

$\hat \pi_j-B_j\mathrm{MMD}(P_S,P_T)\le\alpha_j^\star\le\hat \pi_j+B_j\mathrm{MMD}(P_S,P_T).$

Since accuracy belongs to the interval $[0,1]$, define

$L_j=\max\{0,\hat \pi_j-B_j\mathrm{MMD}(P_S,P_T)\}$

and

$U_j=\min\{1,\hat \pi_j+B_j\mathrm{MMD}(P_S,P_T)\}.$

Then

$L_j\le\alpha_j^\star\le U_j.$

This result is deterministic under Assumption 1.

No confidence parameter $\delta$ is required because both $P_S$ and $P_T$ are defined as the fixed empirical distributions of the selected source set and target workload.

### 14.9 Why subset selection tightens the bound

The width of the source-to-target uncertainty region is determined by

$B_j\mathrm{MMD}(P_S,P_T).$

Therefore, when subset selection produces a smaller MMD,

$\mathrm{MMD}(P_S,P_T)\downarrow,$

the lower bound increases:

$L_j\uparrow.$

At the same time, the upper bound decreases:

$U_j\downarrow.$

Hence, the interval becomes tighter.

In the ideal case,

$\mathrm{MMD}(P_S,P_T)=0.$

Then the theorem gives

$\alpha_j^\star=\hat \pi_j.$

Thus, under the RKHS assumption, if the selected source and target empirical distributions have identical kernel mean embeddings, the source prior accuracy and target accuracy are equal for the correctness function represented in the RKHS.

The facility-location term does not explicitly appear in the theorem.

Its role is complementary to MMD.

MMD controls the average RKHS discrepancy between the source and target distributions, while facility-location coverage encourages the selected subsets to represent target regions that may otherwise be poorly covered.

Therefore, coverage improves representation of the target support, whereas MMD is the quantity directly controlling the theoretical accuracy-shift bound.

### 14.10 Numerical example

Suppose model $F_j$ obtains prior accuracy

$\hat \pi_j=0.84$

on the selected source subsets.

Suppose

$B_j=1.5$

and

$\mathrm{MMD}(P_S,P_T)=0.02.$

The distribution-shift penalty is

$B_j\mathrm{MMD}(P_S,P_T)=1.5\times0.02.$

Therefore,

$B_j\mathrm{MMD}(P_S,P_T)=0.03.$

The lower target-accuracy bound is

$L_j=0.84-0.03.$

Hence,

$L_j=0.81.$

The upper bound is

$U_j=0.84+0.03.$

Hence,

$U_j=0.87.$

Therefore,

$0.81\le\alpha_j^\star\le0.87.$

Under Assumption 1, the target accuracy must lie between $81\%$ and $87\%$.

### 14.11 Optional probabilistic extension when the prior is estimated from a random sample

The previous theorem assumes that $\hat \pi_j$ is the accuracy computed over the complete selected source set.

If instead the selected source distribution contains many items but only $m$ randomly sampled source items are evaluated, then the measured prior itself contains sampling uncertainty.

Let the randomly evaluated source items be

$R_1,\ldots,R_m.$

Let their empirical accuracy be

$\tilde \pi_j=\frac{1}{m}\sum_{r=1}^{m}f_j(R_r).$

Let $\pi_j$ denote the expected accuracy under the selected source distribution.

Then

$\pi_j=E_{a\sim P_S}[f_j(a)].$

Assume that $R_1,\ldots,R_m$ are sampled independently from $P_S$ and that

$0\le f_j(R_r)\le1.$

Hoeffding's inequality gives

$P(\tilde \pi_j-\pi_j\ge\epsilon)\le\exp(-2m\epsilon^2).$

Choose a failure probability $\delta\in(0,1)$.

Set

$\exp(-2m\epsilon^2)=\delta.$

Taking logarithms gives

$-2m\epsilon^2=\log\delta.$

Therefore,

$2m\epsilon^2=\log(1/\delta).$

Hence,

$\epsilon=\sqrt{\frac{\log(1/\delta)}{2m}}.$

It follows that

$P\left(\pi_j\ge\tilde \pi_j-\sqrt{\frac{\log(1/\delta)}{2m}}\right)\ge1-\delta.$

From the MMD result,

$\alpha_j^\star\ge\pi_j-B_j\mathrm{MMD}(P_S,P_T).$

Therefore, with probability at least $1-\delta$,

$\alpha_j^\star\ge\tilde \pi_j-\sqrt{\frac{\log(1/\delta)}{2m}}-B_j\mathrm{MMD}(P_S,P_T).$

Define

$L_j(\delta)=\max\{0,\tilde \pi_j-\sqrt{\frac{\log(1/\delta)}{2m}}-B_j\mathrm{MMD}(P_S,P_T)\}.$

Then

$P(\alpha_j^\star\ge L_j(\delta))\ge1-\delta.$

Similarly, Hoeffding gives

$P\left(\pi_j\le\tilde \pi_j+\sqrt{\frac{\log(1/\delta)}{2m}}\right)\ge1-\delta.$

Combining this with the MMD upper bound gives

$P\left(\alpha_j^\star\le\tilde \pi_j+\sqrt{\frac{\log(1/\delta)}{2m}}+B_j\mathrm{MMD}(P_S,P_T)\right)\ge1-\delta.$

Define

$U_j(\delta)=\min\{1,\tilde \pi_j+\sqrt{\frac{\log(1/\delta)}{2m}}+B_j\mathrm{MMD}(P_S,P_T)\}.$

Then

$P(\alpha_j^\star\le U_j(\delta))\ge1-\delta.$

Therefore, if the prior is itself estimated from a random subset, the lower bound consists of two penalties:

$L_j(\delta)=\tilde \pi_j-\epsilon_{\mathrm{sample}}-\epsilon_{\mathrm{shift}},$

where

$\epsilon_{\mathrm{sample}}=\sqrt{\frac{\log(1/\delta)}{2m}}$

and

$\epsilon_{\mathrm{shift}}=B_j\mathrm{MMD}(P_S,P_T).$

The first term accounts for finite source evaluation.

The second term accounts for source-target distribution shift.

### 14.12 Interpretation of the probabilistic bound

For example, choose

$\delta=0.05.$

Then

$1-\delta=0.95.$

Suppose the lower bound calculated above is

$L_j(0.05)=0.78.$

Then

$P(\alpha_j^\star\ge0.78)\ge0.95.$

The correct interpretation is that the procedure provides a $95\%$ confidence lower bound of $78\%$ for the target accuracy under the stated assumptions.

This probability statement is needed only when there is a random finite-sample component in the source prior.

When the prior accuracy is computed over the entire fixed selected source set and the target of interest is the entire fixed target workload, the RKHS-MMD result in Theorem 1 is deterministic.

### 14.13 Relation to the collision-aware EM prior

The collision-aware EM formulation uses the source-derived prior accuracy in

$\alpha_j\sim\mathrm{Beta}(1+s_j\pi_j,\ 1+s_j(1-\pi_j)).$

In the current pipeline, the prior center can be set to the measured accuracy on the distribution-aligned source set:

$\pi_j=\hat \pi_j.$

The theorem above provides more information than the point estimate alone.

It gives a lower target-aware bound

$L_j$

and an upper target-aware bound

$U_j.$

Therefore, the selected-source prior is accompanied by an explicit certificate describing how much model accuracy can change when moving from the selected source distribution to the target workload.

The interval becomes tighter when

$\mathrm{MMD}(P_S,P_T)$

becomes smaller.

This provides a direct theoretical motivation for the MMD component of the source-subset selection procedure.

### 14.14 Scope of the guarantee

The result in this section concerns the source-derived prior before the target-domain EM updates are applied.

It establishes a relation between

$\hat \pi_j$

and the unknown target accuracy

$\alpha_j^\star.$

In the complete-source evaluation setting, the result is

$L_j\le\alpha_j^\star\le U_j.$

In the random-source-subsample setting, the result is

$P(\alpha_j^\star\ge L_j(\delta))\ge1-\delta.$

This result does not yet establish a corresponding confidence guarantee for the final EM estimate.

The collision-aware EM procedure subsequently uses the target agreement observations $C_i^j$, the latent correctness probabilities $\tau_i^j$, the pseudo-label accuracy $\beta$, and the collision probabilities $\gamma_g$.

A post-EM guarantee requires an additional analysis of these quantities.

The present theorem has a narrower purpose: it proves that the distribution-aligned source prior itself can be bounded relative to the unknown target accuracy using the same RKHS and MMD already present in the subset-selection procedure.

### Bernstein alternative to the Hoeffding bound

Hoeffding's inequality uses only the fact that the correctness observations are bounded between zero and one. It does not use their variance.

When the correctness observations have small variance, Bernstein's inequality can provide a tighter confidence bound.

Suppose model $F_j$ is evaluated on $n$ independently sampled source items $Z_1,\ldots,Z_n$ from the selected source distribution $P_S$.

Let

$Y_j(Z_r)\in\{0,1\}$

denote whether model $F_j$ is correct on source item $Z_r$.

Let the expected source accuracy be

$\pi_j=E_{a\sim P_S}[Y_j(a)].$

The empirical source accuracy is

$\hat \pi_j=\frac{1}{n}\sum_{r=1}^{n}Y_j(Z_r).$

Define the variance of the correctness variable as

$\sigma_j^2=E_{a\sim P_S}[(Y_j(a)-\pi_j)^2].$

Because $Y_j(a)$ is binary,

$\sigma_j^2=\pi_j(1-\pi_j).$

Bernstein's inequality gives, for every $\epsilon>0$,

$P(\hat \pi_j-\pi_j\ge\epsilon)\le\exp\left(-\frac{n\epsilon^2}{2\sigma_j^2+2\epsilon/3}\right).$

An equivalent high-probability form is obtained by choosing a failure probability $\delta\in(0,1)$.

With probability at least $1-\delta$,

$\hat \pi_j-\pi_j\le\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}+\frac{\log(1/\delta)}{3n}.$

Rearranging gives a lower confidence bound for the expected source accuracy:

$\pi_j\ge\hat \pi_j-\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}-\frac{\log(1/\delta)}{3n}.$

Therefore,

$P\left(\pi_j\ge\hat \pi_j-\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}-\frac{\log(1/\delta)}{3n}\right)\ge1-\delta.$

We now combine this sampling bound with the RKHS-MMD distribution-shift bound derived previously.

From the MMD result,

$\alpha_j^\star\ge\pi_j-B_j\mathrm{MMD}(P_S,P_T).$

Whenever the Bernstein event holds,

$\alpha_j^\star\ge\hat \pi_j-\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}-\frac{\log(1/\delta)}{3n}-B_j\mathrm{MMD}(P_S,P_T).$

Therefore,

$P\left(\alpha_j^\star\ge\hat \pi_j-\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}-\frac{\log(1/\delta)}{3n}-B_j\mathrm{MMD}(P_S,P_T)\right)\ge1-\delta.$

Define the Bernstein lower bound as

$L_j^{B}(\delta)=\max\left\{0,\hat \pi_j-\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}-\frac{\log(1/\delta)}{3n}-B_j\mathrm{MMD}(P_S,P_T)\right\}.$

Then,

$P(\alpha_j^\star\ge L_j^{B}(\delta))\ge1-\delta.$

The same reasoning gives an upper bound.

Bernstein's inequality implies that, with probability at least $1-\delta$,

$\pi_j\le\hat \pi_j+\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}+\frac{\log(1/\delta)}{3n}.$

Using the MMD upper bound

$\alpha_j^\star\le\pi_j+B_j\mathrm{MMD}(P_S,P_T),$

we obtain

$\alpha_j^\star\le\hat \pi_j+\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}+\frac{\log(1/\delta)}{3n}+B_j\mathrm{MMD}(P_S,P_T).$

Define

$U_j^{B}(\delta)=\min\left\{1,\hat \pi_j+\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}+\frac{\log(1/\delta)}{3n}+B_j\mathrm{MMD}(P_S,P_T)\right\}.$

Then,

$P(\alpha_j^\star\le U_j^{B}(\delta))\ge1-\delta.$

Thus, Bernstein gives the target-accuracy interval

$L_j^{B}(\delta)\le\alpha_j^\star\le U_j^{B}(\delta)$

with the corresponding one-sided confidence guarantees.

#### Comparison with Hoeffding

The Hoeffding lower bound is

$L_j^{H}(\delta)=\max\left\{0,\hat \pi_j-\sqrt{\frac{\log(1/\delta)}{2n}}-B_j\mathrm{MMD}(P_S,P_T)\right\}.$

The Bernstein lower bound is

$L_j^{B}(\delta)=\max\left\{0,\hat \pi_j-\sqrt{\frac{2\sigma_j^2\log(1/\delta)}{n}}-\frac{\log(1/\delta)}{3n}-B_j\mathrm{MMD}(P_S,P_T)\right\}.$

Hoeffding requires only

$0\le Y_j(Z_r)\le1.$

Bernstein additionally uses the variance $\sigma_j^2$.

For binary correctness,

$\sigma_j^2=\pi_j(1-\pi_j)\le\frac{1}{4}.$

When $\sigma_j^2$ is substantially smaller than $1/4$, the Bernstein sampling penalty can be smaller than the Hoeffding sampling penalty.

Therefore, Bernstein can provide a tighter target-accuracy certificate when the correctness behavior of the model has low variance.

#### Unknown variance

The ordinary Bernstein bound above assumes that $\sigma_j^2$ is known or that a valid upper bound on it is available.

Because $\pi_j$ is unknown in practice, the exact quantity

$\sigma_j^2=\pi_j(1-\pi_j)$

is also unknown.

A universally valid upper bound for binary correctness is

$\sigma_j^2\le\frac{1}{4}.$

Substituting this bound gives the conservative Bernstein certificate

$L_j^{B,\mathrm{cons}}(\delta)=\max\left\{0,\hat \pi_j-\sqrt{\frac{\log(1/\delta)}{2n}}-\frac{\log(1/\delta)}{3n}-B_j\mathrm{MMD}(P_S,P_T)\right\}.$

However, this conservative form may not improve over Hoeffding because it does not exploit the actual variance.

If the variance is estimated from the observed source sample, then a separate empirical-Bernstein inequality should be used rather than directly substituting the empirical variance into the ordinary Bernstein formula.

Thus, the two bounds serve different purposes:

- Hoeffding provides a simple variance-free certificate.
- Bernstein provides a potentially tighter certificate when the variance is known or can be bounded sharply.
- Empirical Bernstein is appropriate when the variance itself must be estimated from the observed data.

