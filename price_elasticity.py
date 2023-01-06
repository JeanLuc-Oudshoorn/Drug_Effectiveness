import bambi as bmb
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data and add month variable
avocado = pd.read_csv("avocado.csv")
avocado['month'] = pd.DatetimeIndex(avocado['Date']).month

# Filter for conventional avocados
avocado = avocado[avocado.type == "conventional"]

# Plot seasonality
# (No clear repeating seasonal trend)
tmp = avocado[['Date', 'AveragePrice', 'Total Volume']]
seasonality = tmp.groupby("Date").mean(["AveragePrice", "Total Volume"])
seasonality = seasonality.reset_index()

fig, ax = plt.subplots()
ax.plot(seasonality['Date'], seasonality['AveragePrice'], color="red")
ax2 = ax.twinx()
ax2.plot(seasonality['Date'], seasonality['Total Volume'], color="orange")
plt.xticks(np.arange(0, 169, step=39))
plt.title("Development of Price and Volume over Time")
plt.savefig("weekly_price.png")
plt.show()
plt.clf()

# Calculate averages for regional markets and month of each year
regional_sub = avocado[['AveragePrice', 'Total Volume', 'month', 'year', 'region']]
regional_avg = regional_sub.groupby(['region', 'year', 'month']).mean(['AveragePrice', 'Total Volume']).reset_index()
regional_avg = regional_avg.rename(columns={'AveragePrice': 'AdjPrice', 'Total Volume': 'AdjVolume'})

# Adjust original data for regional averages
avocado_adj = avocado.merge(regional_avg, on=['region', 'year', 'month'], how='left')
avocado_adj['PricePerc'] = (avocado_adj['AveragePrice'] / avocado_adj['AdjPrice'] - 1)*100
avocado_adj['VolPerc'] = (avocado_adj['Total Volume'] / avocado_adj['AdjVolume'] - 1)*100
avocado_adj = avocado_adj[['Date', 'PricePerc', 'VolPerc', 'region']]
g, s, i = np.polyfit(avocado_adj['PricePerc'], avocado_adj['VolPerc'], 2)

# Plot variation in average price and volume
plt.grid(visible=True)
plt.plot(avocado_adj['PricePerc'], avocado_adj['VolPerc'], '.', alpha=0.25)
x_var = np.linspace(-40, 40, num=200)
plt.plot(x_var, g*x_var**2 + s*x_var + i)
plt.title("Percentual Change in Price versus percentual Change in Volume")
plt.xlabel("Change in Price (%)")
plt.ylabel("Change in Volume (%)")
plt.savefig('price_elasticity_scatter.png')
plt.show()
plt.clf()

# Define Bayesian price elasticity model
avocado_adj['PricePercSq'] = avocado_adj['PricePerc']**2

model = bmb.Model('VolPerc ~ PricePerc + PricePercSq', avocado_adj)
result = model.fit(draws=1000, chains=4, cores=1)

az.plot_trace(result)
plt.show()
plt.clf()
print(az.summary(result))

# Extracting posterior samples
priceperc_samples = np.array(result['posterior']['PricePerc']).reshape(-1)
pricepercsq_samples = np.array(result['posterior']['PricePercSq']).reshape(-1)
intercept = np.array(result['posterior']['Intercept']).reshape(-1)

# Plotting potential Regression Lines based on sample Parameters
plt.grid(visible=True)
plt.plot(avocado_adj['PricePerc'], avocado_adj['VolPerc'], '.', alpha=0.2)
for i in range(500):
    _ = plt.plot(x_var, pricepercsq_samples[i]*x_var**2 + priceperc_samples[i]*x_var + intercept[i], alpha=0.3,
                 linewidth=0.3, color='red')
plt.title("Avocado Price Elasticity")
plt.xlabel("Change in Price (%)")
plt.ylabel("Change in Volume (%)")
plt.savefig('price_elasticity_fit.png', dpi=150)
plt.show()
