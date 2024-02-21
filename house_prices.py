# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in Dutch house price index data
house_prices = pd.read_csv('Observations.csv', sep=';')

# Filter out index values
house_prices = house_prices[house_prices['Measure'] == 'M001505_2']

# Keep only monthly observations
house_prices = house_prices[house_prices['Perioden'].str.contains('MM')]

# Extract year from data
house_prices['Year'] = house_prices['Perioden'].str.extract(r'([0-9]{4})').astype('int')

# Extract month from data
house_prices['Month'] = house_prices['Perioden'].str.extract(r'[0-9]{4}MM([0-9]{2})').astype('int')

# Create date
house_prices['Date'] = pd.to_datetime(house_prices['Year']*10000 + house_prices['Month']*100 + 1, format="%Y%m%d")

# Convert comma into dot and convert datatype to float
house_prices['HouseIDX'] = house_prices['Value'].str.replace(',', '.').astype('float')

# Save house price actuals
raw_house_prices = house_prices

# Drop unnecessary columns
house_prices.drop(columns=['Id', 'Measure', 'Value', 'Perioden', 'ValueAttribute', 'Year', 'Month'], inplace=True)

# Set date as index
house_prices.set_index('Date', inplace=True)

# Subset for comparison with other financial time series
house_prices = house_prices.loc['2013-07-01':]

# Plot the result
house_prices.plot()
plt.title("Dutch House price development")
plt.ylabel("Index Value")
plt.xlabel("Year")
plt.savefig('dutch_house_prices.png')
plt.show()

# Log transform value column
house_prices['HouseIDX'] = np.log(house_prices['HouseIDX'].dropna()).diff()

# Save result
house_prices.to_csv('house_prices.csv')
