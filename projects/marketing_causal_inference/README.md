# From Prediction to Causation: Measuring Incremental Marketing Impact

**Status:** Research starter, synthetic simulation. Not a publication or an empirical result about any real company.

## Motivation
A marketing model can correctly predict who is likely to purchase yet fail to identify whose behavior *changes* because of an advertisement. That gap matters for resource allocation and a manager's return on marketing spend.

## Research questions
1. How different are customer purchase propensity and customer-level treatment effects?
2. Does allocating limited campaign impressions using estimated heterogeneous treatment effects outperform targeting by purchase propensity in a synthetic randomized setting?
3. How sensitive are estimated gains to sample size, uncertainty, and campaign cost?

## Design
A reproducible synthetic randomized experiment where treatment assignment is independent of pre-treatment features. A held-out set evaluates policy performance. Generate outcomes from a known data-generating process, fit separate models by treatment arm, and compare a policy targeting high estimated uplift to a policy targeting high baseline response likelihood.

## Estimands
- Average treatment effect (ATE): E[Y(1)-Y(0)].
- Conditional average treatment effect (CATE): E[Y(1)-Y(0) | X].
- Policy value for each targeting strategy, evaluated on a fresh simulated sample whose potential outcomes are known (synthetic benchmark only).

## Hypotheses
- H1: Higher predicted baseline purchase probability does not necessarily imply higher incremental response.
- H2: Targeting based on well-estimated uplift improves policy value when effects are heterogeneous, compared with propensity targeting, under a fixed targeting budget.

## Run
```bash
pip install -r requirements.txt
python experiment.py
```
This writes `results/summary.csv`, `results/policy_comparison.png`, and `results/ate_bootstrap.png`. The results directory is ignored from version control by default.

## Next steps for a PhD-ready study
- Find appropriately licensed real experimental data; record source, selection, and limitations.
- Pre-register research questions, outcomes, and inclusion/exclusion criteria.
- Add baseline difference tests, uncertainty estimates for policy differences, repeated splits, and stronger uplift estimators.
- Compare against an observational setting and discuss identification assumptions.
- Produce a structured research paper with literature review, methodology, robustness checks, and limitations.

## Ethical boundary
Do not publish confidential employer/client datasets or claim these synthetic outputs are real marketing gains.
