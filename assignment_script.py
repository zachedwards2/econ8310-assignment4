import pandas as pd
import numpy as np
import pymc as pm
import scipy.special as sp
import requests
from io import BytesIO

# data
url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv"
r = requests.get(url, stream=True)
df = pd.read_csv(BytesIO(r.content))


df['treatment'] = (df['version'] == 'gate_40').astype(int)

def run_model(retention_col):
    df['ret'] = df[retention_col].astype(int)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1.5)
        beta  = pm.Normal("beta", mu=0, sigma=1.5)

        logit_p = alpha + beta * df['treatment'].values
        
        retention = pm.Bernoulli("retention", 
                                 logit_p=logit_p,
                                 observed=df['ret'].values)

        trace = pm.sample(2000, tune=2000, target_accept=0.95, progressbar=True)

    # posteriors
    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta_samples  = trace.posterior['beta'].values.flatten()

    p30 = sp.expit(alpha_samples)
    p40 = sp.expit(alpha_samples + beta_samples)
    diff = p40 - p30
    P_treat_worse = (diff < 0).mean()

    return trace, p30, p40, diff, P_treat_worse


# models

trace_1, p30_1, p40_1, diff_1, prob_1 = run_model('retention_1')


trace_7, p30_7, p40_7, diff_7, prob_7 = run_model('retention_7')


#results
print("1 daty")
print("Posterior mean retention (gate_30):", p30_1.mean())
print("Posterior mean retention (gate_40):", p40_1.mean())
print("Difference (gate_40 − gate_30):", diff_1.mean())
print("Posterior probability gate_40 < gate_30:", prob_1)

print("7 day")
print("Posterior mean retention (gate_30):", p30_7.mean())
print("Posterior mean retention (gate_40):", p40_7.mean())
print("Difference (gate_40 − gate_30):", diff_7.mean())
print("Posterior probability gate_40 < gate_30:", prob_7)


#Had a plotly chart but didn't like it
