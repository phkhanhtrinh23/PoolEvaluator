# Baseline implementations

This package contains the baseline code that existed in the repository before the
paper-artifact cleanup, updated to consume the current `[items, models]` response
layout. `Independent`, `Majority`, `DawidSkene`, `AgreementLine`, and `LLMJudge`
were previously under `baselines/`. DoC, ATC, and GDE were previously grouped in
the archived domain-evaluation module and now live here as first-class baselines.

| Method | Python entry point | Required inputs |
|---|---|---|
| Independent | `Independent.evaluate` | labeled source responses and gold answers |
| Majority | `Majority.evaluate` | unlabeled target responses |
| Dawid--Skene | `DawidSkene.evaluate` | unlabeled target responses |
| Agreement-on-the-Line | `AgreementLine.evaluate` | source responses/labels and target responses |
| LLM-as-judge | `LLMJudge.evaluate` | target responses and a judge callback |
| DoC | `DifferenceOfConfidence.evaluate` | source/target probabilities and source labels |
| ATC-MC / ATC-NE | `AverageThresholdedConfidence.evaluate` | source/target probabilities and source labels |
| GDE | `GeneralizedDisagreementEquality.evaluate` | two independently trained replicas' predictions |

No implementation was found in any Git branch for FusionSQL, MetaEvaluator,
NL2SQL-BUGs, ConfProfile, AutoEval, Discrepancy, GLAD, SpectralDS, LearnCrowds,
or GNNEvaluator. Those methods require their official upstream implementations or a
separate faithful port; they are not represented here by placeholder results.
