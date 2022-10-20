import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Running on PyMC3 v{pm.__version__}")

# Read in pre-scraped data
trader_frame = pd.read_csv("trader_frame.csv")

# Extract list of traders
trader_list = list(trader_frame.columns)[1:-1]

# Create dictionary to save posterior samples
sample_dict = dict()


for trader in trader_list:
    # Extract Outcomes
    y = np.array((trader_frame[trader] - trader_frame['SP500']).dropna())

    mu_mean = 0
    mu_sd = np.std(y) * 2

    sd_low = np.std(y) * 0.01
    sd_high = np.std(y) * 100

    with pm.Model() as model:
        # Priors for means by group
        diff_means = pm.Normal("diff_means", mu=mu_mean, sigma=mu_sd)

        # Priors for standard deviations by group
        diff_std = pm.Uniform("diff_std", lower=sd_low, upper=sd_high)

        # Prior for nu parameter
        vu = pm.Exponential("vu", 1 / 5.0) + 1

        # Converting standard deviation to precision
        lam = diff_std ** -2

        # Defining the Posterior
        obs = pm.StudentT("obs", nu=vu, mu=diff_means, lam=lam, observed=y)

    if __name__ == "__main__":
        with model:
            # Perform Markov Chain Monte Carlo
            trace = pm.sample(3000, tune=1800, cores=4)

            # Sample from the Posterior
            posterior_sample = pm.sample_posterior_predictive(
                trace, var_names=["diff_means"]
            )

            # Save posterior sample for each trader to dictionary
            sample_dict[trader] = posterior_sample['diff_means']

if len(sample_dict.keys()) == len(trader_list):
    sample_df = pd.DataFrame.from_dict(sample_dict)

    subsample_df = sample_df[['Jeppe Kirk Bonde', 'Jay Edward Smith', 'Blue Screen Media ApS', 'Harry Stephan Harrison',
                              'Libor Vasa', 'Reinhardt Gert Coetzee']]

    sns.set_theme()
    sns.kdeplot(data=subsample_df)
    plt.axvline(x=0, color='black')
    plt.title("Estimated Distribution of Outperformance vs. S&P 500")
    plt.xlabel("Log Mean Outperformance (0.02 ~ 2% per month)")
    plt.savefig('bayesian_density.png')
    plt.show()
