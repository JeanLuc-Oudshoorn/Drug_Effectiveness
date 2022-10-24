import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

print(f"Running on PyMC3 v{pm.__version__}")


# Simulated outcomes
drug = (101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
        109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
        96,103,124,101,101,100,101,101,104,100,101,100,99,100,104)

placebo = (99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
           104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
           101,100,99,101,100,102,99,100,99,102,100,100,105)


# Put into numpy arrays
y1 = np.array(drug)

y2 = np.array(placebo)
y = pd.DataFrame(
    dict(value=np.r_[y1, y2], group=np.r_[["drug"] * len(drug), ["placebo"] * len(placebo)])
)


μ_m = y.value.mean()
μ_s = y.value.std() * 2


with pm.Model() as model:
    # Priors for means by group
    group1_mean = pm.Normal("group1_mean", mu=μ_m, sigma=μ_s)
    group2_mean = pm.Normal("group2_mean", mu=μ_m, sigma=μ_s)

σ_low = 1
σ_high = 10

with model:
    # Priors for standard deviations by group
    group1_std = pm.Uniform("group1_std", lower=σ_low, upper=σ_high)
    group2_std = pm.Uniform("group2_std", lower=σ_low, upper=σ_high)


with model:
    # Prior for nu parameter (assumed to apply to both groups)
    ν = pm.Exponential("ν_minus_one", 1 / 29.0) + 1

    # Converting standard deviation to precision
    λ1 = group1_std ** -2
    λ2 = group2_std ** -2

    # Defining the Posterior
    group1 = pm.StudentT("drug", nu=ν, mu=group1_mean, lam=λ1, observed=y1)
    group2 = pm.StudentT("placebo", nu=ν, mu=group2_mean, lam=λ2, observed=y2)

with model:
    # Extra Metrics: Difference of Posterior Means
    diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
    diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
    effect_size = pm.Deterministic(
        "effect size", diff_of_means / np.sqrt((group1_std ** 2 + group2_std ** 2) / 2)
    )

if __name__ == "__main__":
    with model:
        trace = pm.sample(3000, tune=2000, cores=4)

        az.plot_posterior(
            trace,
            var_names=["difference of means", "effect size"],
            ref_val=0,
            color="#87ceeb")
        plt.savefig('difference_of_means.png')
        plt.show()

        print(az.summary(trace, var_names=["difference of means", "difference of stds", "effect size"]))

