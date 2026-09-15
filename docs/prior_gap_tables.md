# Prior gap, model by model

For each model two accuracies, in **accuracy percentage points**:

- **labeled** -- its accuracy on the labeled split (what the prior tells us)
- **target** -- its true accuracy on the unlabeled target pool (what we actually want)

and the signed gap between them, `error = labeled - target`. A positive error means the prior is *too optimistic* about that model.

The three summary rows separate the two kinds of mistake a prior can make:

| row | formula | question it answers |
|---|---|---|
| `bias` | mean of the signed errors | is the prior wrong in the same direction for every model? |
| `MAE` | mean of `|error|` | how wrong is the prior overall? (this number *contains* the bias) |
| `cMAE` | mean of `|error - bias|` | once the shared offset is removed, does the prior still rank the models correctly? |

`bias == MAE` means every single model is overstated, by nearly the same amount -- a pure **level** error, which a handful of target labels can correct. A large `cMAE` means a **shape** error, which they cannot.

## Text-to-SQL

| model | labeled (Spider) | target (Spider) | error | labeled (BIRD) | target (BIRD) | error |
|---|---|---|---|---|---|---|
| gpt4o-schema | 80.8 | 76.0 | +4.8 | 43.3 | 40.7 | +2.7 |
| gpt4o-minimal | 77.5 | 76.0 | +1.5 | 39.2 | 34.7 | +4.5 |
| gpt4omini-schema | 76.7 | 72.7 | +4.0 | 38.3 | 35.3 | +3.0 |
| gpt4omini-fewshot | 80.0 | 73.3 | +6.7 | 40.0 | 36.0 | +4.0 |
| gpt41 | 76.7 | 76.7 | +0.0 | 42.5 | 43.3 | -0.8 |
| gpt41-mini | 77.5 | 80.0 | -2.5 | 44.2 | 42.7 | +1.5 |
| gpt41-nano | 72.5 | 73.3 | -0.8 | 36.7 | 28.7 | +8.0 |
| gpt4-turbo | 79.2 | 76.0 | +3.2 | 40.0 | 37.3 | +2.7 |
| gpt35-schema | 79.2 | 69.3 | +9.8 | 37.5 | 33.3 | +4.2 |
| gpt35-fewshot | 73.3 | 69.3 | +4.0 | 41.7 | 36.7 | +5.0 |
| **bias** |  |  | **+3.07** |  |  | **+3.47** |
| **MAE** |  |  | **3.73** |  |  | **3.63** |
| **cMAE** |  |  | **2.82** |  |  | **1.67** |

## Image classification

| model | labeled (MNIST->USPS) | target (MNIST->USPS) | error | labeled (MNIST->SVHN) | target (MNIST->SVHN) | error |
|---|---|---|---|---|---|---|
| ResNeXt50-s0 | 99.6 | 64.7 | +34.9 | 99.7 | 6.3 | +93.4 |
| ResNeXt50-s1 | 99.6 | 34.0 | +65.5 | 99.6 | 6.6 | +93.0 |
| ResNeXt50-s2 | 99.6 | 46.4 | +53.2 | 99.6 | 6.3 | +93.3 |
| RegNetY8GF-s0 | 99.6 | 43.5 | +56.1 | 99.6 | 6.2 | +93.5 |
| RegNetY8GF-s1 | 99.5 | 62.6 | +36.9 | 99.5 | 6.2 | +93.4 |
| RegNetY8GF-s2 | 99.6 | 34.9 | +64.7 | 99.6 | 7.1 | +92.5 |
| ConvNeXtT-s0 | 99.0 | 89.4 | +9.6 | 99.0 | 33.0 | +66.0 |
| ConvNeXtT-s1 | 99.0 | 90.5 | +8.5 | 99.0 | 35.5 | +63.5 |
| ConvNeXtT-s2 | 99.0 | 89.0 | +9.9 | 99.0 | 33.1 | +65.9 |
| ViT-Tiny-s0 | 98.9 | 78.4 | +20.5 | 98.9 | 9.4 | +89.5 |
| ViT-Tiny-s1 | 98.9 | 71.8 | +27.1 | 98.9 | 8.5 | +90.4 |
| ViT-Tiny-s2 | 98.7 | 84.4 | +14.4 | 98.7 | 11.0 | +87.8 |
| DeiT-Small-s0 | 98.6 | 87.3 | +11.3 | 98.6 | 11.2 | +87.4 |
| DeiT-Small-s1 | 98.8 | 85.9 | +12.9 | 98.8 | 10.4 | +88.4 |
| DeiT-Small-s2 | 98.5 | 82.8 | +15.7 | 98.5 | 13.6 | +84.8 |
| **bias** |  |  | **+29.42** |  |  | **+85.51** |
| **MAE** |  |  | **29.42** |  |  | **85.51** |
| **cMAE** |  |  | **17.97** |  |  | **8.25** |

## Node classification

| model | labeled (A->C) | target (A->C) | error | labeled (D->A) | target (D->A) | error |
|---|---|---|---|---|---|---|
| GCN-s0 | 80.7 | 72.6 | +8.1 | 80.9 | 59.9 | +21.0 |
| GCN-s1 | 81.6 | 70.9 | +10.7 | 81.8 | 60.4 | +21.4 |
| GCN-s2 | 81.3 | 70.8 | +10.5 | 80.8 | 60.0 | +20.8 |
| SAGE-s0 | 80.7 | 70.0 | +10.7 | 80.6 | 57.1 | +23.5 |
| SAGE-s1 | 80.3 | 70.5 | +9.8 | 80.3 | 56.4 | +23.9 |
| SAGE-s2 | 81.1 | 72.0 | +9.1 | 80.6 | 57.0 | +23.6 |
| GAT-s0 | 79.9 | 68.7 | +11.2 | 80.9 | 59.2 | +21.7 |
| GAT-s1 | 81.9 | 67.1 | +14.9 | 81.2 | 57.1 | +24.1 |
| GAT-s2 | 80.6 | 68.0 | +12.6 | 81.8 | 57.8 | +23.9 |
| GIN-s0 | 29.6 | 26.0 | +3.6 | 31.5 | 29.6 | +2.0 |
| GIN-s1 | 29.6 | 26.0 | +3.6 | 75.5 | 54.0 | +21.4 |
| GIN-s2 | 80.4 | 65.4 | +15.0 | 79.9 | 56.0 | +23.8 |
| MLP-s0 | 52.1 | 52.2 | -0.1 | 55.6 | 46.3 | +9.3 |
| MLP-s1 | 53.1 | 52.4 | +0.7 | 56.3 | 46.7 | +9.6 |
| MLP-s2 | 52.4 | 52.4 | -0.0 | 57.1 | 46.4 | +10.6 |
| **bias** |  |  | **+8.02** |  |  | **+18.73** |
| **MAE** |  |  | **8.03** |  |  | **18.73** |
| **cMAE** |  |  | **4.30** |  |  | **5.78** |

## What the three modalities show

Read the `error` columns first. In all six datasets almost every entry is **positive**:
the labeled split says every model is better than it really is on the target pool. That is
why `bias` and `MAE` are nearly identical everywhere -- nothing cancels.

| modality | MAE | bias | cMAE | reading |
|---|---|---|---|---|
| Text-to-SQL (Spider) | 3.73 | +3.07 | 2.82 | small gap; the prior is nearly usable as-is |
| Text-to-SQL (BIRD) | 3.63 | +3.47 | 1.67 | small gap, and the shape is very good |
| Node class. (A->C) | 8.03 | +8.02 | 4.30 | level error; shape survives |
| Node class. (D->A) | 18.73 | +18.73 | 5.78 | large level error, shape still good |
| Image class. (MNIST->USPS) | 29.42 | +29.42 | 17.97 | large in BOTH level and shape |
| Image class. (MNIST->SVHN) | **85.51** | +85.51 | 8.25 | catastrophic level, surprisingly good shape |

**MNIST->SVHN is the extreme case and the clearest one.** Every model scores 98-100 on the
labeled split and 6-36 on the target. The gap is ~85 points for *everyone*, so `bias` is
85.51 and `cMAE` only 8.25. The prior is not confused about which model is better -- it is
uniformly wrong about how hard the task is. One scalar correction would fix most of it.

**MNIST->USPS is the harder failure**, even though its MAE is three times smaller. Its
`cMAE` is 17.97, the largest in the table, because the models do *not* degrade together:
ConvNeXt drops ~9 points while ResNeXt-s1 drops 65. No scalar correction can repair that;
the prior genuinely does not know which model will transfer. This is why the retrieval
experiment found the vision priors rank models in reverse (`rho` -0.62, -0.63; see
[retrieval_prior.md](retrieval_prior.md)).

**Text-to-SQL is the benign case.** Errors run -2.5 to +9.8 on Spider, and `cMAE` of 2.82
and 1.67 say the prior's ranking is close to right. The labeled split and the target pool
are the same kind of data, so there is no domain gap to speak of.

**Node classification sits in between**, and shows one useful detail: the degenerate models
(GIN-s0/s1 stuck at ~29, MLPs at ~52) have small errors simply because they had no height
to fall from. They are what keeps `cMAE` from being near zero.

## Why this decomposition drives the method

A **level** error is cheap to fix: a handful of validated target items tells you the whole
pool is ~19 points lower than the prior claims. A **shape** error is not -- no amount of
scalar correction recovers a ranking that is wrong.

Five of six datasets here are dominated by the level error. That is the justification for
spending expert budget on target labels rather than on better priors, and it matches
[random_init.md](random_init.md): the alpha anchor is worth ~9.8 accuracy points with no
validation but only ~2.7 after 40 expert labels, because validation supplies exactly the
level that the prior gets wrong.

## Reproduce

    python experiments/report_prior_gap.py
