import pandas as pd
import numpy as np
import bambi as bmb
import matplotlib.pyplot as plt

# Read in trader data
trader_frame = pd.read_csv("trader_frame_upd.csv")

# Generate squared S&P 500 values
trader_frame['SQ'] = trader_frame['^GSPC']**2
trader_frame['SP'] = trader_frame['^GSPC']

# Rename columns for formula notation
trader_frame.rename(columns={"Jeppe Kirk Bonde": "Jeppe",
                             "Harry Stephan Harrison": "Harry"},
                    inplace=True)

# Create first trader model
jep_model = bmb.Model('Jeppe ~ SP + SQ', trader_frame[['Jeppe', 'SP', 'SQ']].dropna())
jep_result = jep_model.fit(draws=1000, chains=4, cores=1)

# Extracting posterior samples for first trader
jep2_samples = np.array(jep_result['posterior']['SQ']).reshape(-1)
jep1_samples = np.array(jep_result['posterior']['SP']).reshape(-1)
jep0_samples = np.array(jep_result['posterior']['Intercept']).reshape(-1)


# Create second trader model
cph_model = bmb.Model('ASML ~ SP + SQ', trader_frame[['ASML', 'SP', 'SQ']].dropna())
cph_result = cph_model.fit(draws=1000, chains=4, cores=1)

# Extracting posterior samples for first trader
cph2_samples = np.array(cph_result['posterior']['SQ']).reshape(-1)
cph1_samples = np.array(cph_result['posterior']['SP']).reshape(-1)
cph0_samples = np.array(cph_result['posterior']['Intercept']).reshape(-1)


# Create third trader model
har_model = bmb.Model('Harry ~ SP + SQ', trader_frame[['Harry', 'SP', 'SQ']].dropna())
har_result = har_model.fit(draws=1000, chains=4, cores=1)

# Extracting posterior samples for first trader
har2_samples = np.array(har_result['posterior']['SQ']).reshape(-1)
har1_samples = np.array(har_result['posterior']['SP']).reshape(-1)
har0_samples = np.array(har_result['posterior']['Intercept']).reshape(-1)

# Define X variable
x_var = np.linspace(start=-0.2, stop=0.2, num=100)


# Plot trader relations to S&P 500
fig, ax = plt.subplots()
plt.grid(visible=True)
for i in range(250):
    _ = plt.plot(x_var, jep2_samples[i]*x_var**2 + jep1_samples[i]*x_var + jep0_samples[i], alpha=0.4,
                 linewidth=0.2, color='red')
    _ = plt.plot(x_var, cph2_samples[i]*x_var**2 + cph1_samples[i]*x_var + cph0_samples[i], alpha=0.4,
                 linewidth=0.2, color='blue')
    _ = plt.plot(x_var, har2_samples[i]*x_var**2 + har1_samples[i]*x_var + har0_samples[i], alpha=0.4,
                 linewidth=0.2, color='green')

plt.legend(['Jeppe Kirk Bonde', 'ASML', 'Harry Stephan Harrison'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='grey')
ax2 = ax.twinx()
ax2.hist(trader_frame['SP'], color='grey', alpha=0.25, bins=20)
plt.suptitle("Trader Performance Correlation to S&P 500")
plt.title("with histogram of S&P 500 performance")
plt.xlim(-0.15, 0.15)
ax.set_ylim(-0.15, 0.15)
ax2.set_ylim(0, 40)
ax.set_xlabel('S&P 500 log performance (0.01 ~ 1%)')
ax.set_ylabel('Trader log performance (0.01 ~ 1%)')
ax2.set_ylabel('Count')
plt.savefig('trader_correlation.png')
plt.show()
