"""Synthetic randomized marketing experiment, for learning/research illustration only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)
N = 15000
BUDGET_FRACTION = 0.30
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def make_data(n, generator):
    x1, x2 = generator.normal(size=(2, n))
    p0 = sigmoid(-1.35 + 1.15*x1 + .65*x2)
    # Incremental effect is driven by x2, not simply by high baseline purchase probability.
    p1 = sigmoid(-1.35 + 1.15*x1 + .65*x2 + .12 + .95*x2)
    t = generator.binomial(1, .5, n)
    y = generator.binomial(1, np.where(t==1, p1, p0))
    X = pd.DataFrame({'feature_1': x1, 'feature_2': x2})
    return X, t, y, p0, p1

X, t, y, p0, p1 = make_data(N, rng)
tr, te = train_test_split(np.arange(N), test_size=.35, random_state=SEED, stratify=t)
models = {}
for arm in (0, 1):
    subset = tr[t[tr]==arm]
    model = LogisticRegression(max_iter=500)
    model.fit(X.iloc[subset], y[subset])
    models[arm] = model
p_control = models[0].predict_proba(X.iloc[te])[:,1]
p_treat = models[1].predict_proba(X.iloc[te])[:,1]
uplift = p_treat - p_control
k = int(len(te)*BUDGET_FRACTION)
policies = {
  'Random': rng.permutation(len(te))[:k],
  'Propensity targeting': np.argsort(-p_control)[:k],
  'Uplift targeting': np.argsort(-uplift)[:k],
}
# These are oracle policy values using known synthetic potential-outcome probabilities.
# Not deployable real-world estimates.
rows=[]
for name, indices in policies.items():
    assigned = np.zeros(len(te), dtype=bool)
    assigned[indices]=True
    policy_value = np.mean(np.where(assigned, p1[te], p0[te]))
    incremental = policy_value-np.mean(p0[te])
    rows.append({'policy':name, 'synthetic_policy_conversion_rate':policy_value,
                 'synthetic_incremental_conversion_points':incremental*100})
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'summary.csv',index=False)
print('Synthetic experiment only — no real company or campaign data')
print(summary.to_string(index=False, float_format=lambda v:f'{v:.4f}'))

fig,ax=plt.subplots(figsize=(8,4.5))
ax.bar(summary['policy'],summary['synthetic_incremental_conversion_points'])
ax.set_ylabel('Incremental conversion (percentage points)')
ax.set_title('Synthetic oracle policy evaluation')
ax.tick_params(axis='x',rotation=12)
fig.tight_layout();fig.savefig(OUT/'policy_comparison.png',dpi=160);plt.close(fig)

# Nonparametric bootstrap of observed randomized ATE on held-out sample.
obs_t = t[te]; obs_y=y[te]
ate=float(obs_y[obs_t==1].mean()-obs_y[obs_t==0].mean())
boot=np.empty(1000)
for b in range(1000):
    idx=rng.integers(0,len(te),len(te))
    bt=obs_t[idx];by=obs_y[idx]
    boot[b]=by[bt==1].mean()-by[bt==0].mean()
ci=np.quantile(boot,[.025,.975]);print(f'Observed holdout ATE: {ate:.4f}; 95% bootstrap CI: [{ci[0]:.4f}, {ci[1]:.4f}]')
fig,ax=plt.subplots(figsize=(7,4));ax.hist(boot,bins=30);ax.axvline(ate,linestyle='--',label='Estimated ATE');ax.set_xlabel('ATE estimate');ax.set_ylabel('Bootstrap frequency');ax.set_title('Synthetic RCT: ATE bootstrap distribution');ax.legend();fig.tight_layout();fig.savefig(OUT/'ate_bootstrap.png',dpi=160);plt.close(fig)
