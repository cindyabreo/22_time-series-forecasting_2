import pandas as pd
from datetime import datetime
import numpy as np

date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')

date_rng

type(date_rng[0])

df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = np.random.randint(0,100,size=(len(date_rng)))

df.head()

#Convert the dataframe index to a datetime index 

df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime')
df.drop(['date'], axis=1, inplace=True)
df.head()

# Example on how to filter data with only day 2.

df[df.index.day == 2]

# Filtering data between two dates

df['2018-01-04':'2018-01-06']

df.resample('D').mean()

# Example on how to get the sum of the last three values.

df['rolling_sum'] = df.rolling(3).sum()
df.head(10)

import matplotlib.pyplot as plt
df['rolling_sum_mean']=df['data'].rolling(4).mean()

df.head(4)

df.columns

df.values

df.values[:,2]

plt.plot(df['data'].values,'r')
plt.plot(df.values[:,2],'b')
plt.show()

df['rolling_sum_backfilled'] = df['rolling_sum'].fillna(method='backfill')
df.head()

epoch_t = 1529272655
real_t = pd.to_datetime(epoch_t, unit='s')
real_t

# Now, let's convert it to Pacific time

real_t.tz_localize('UTC').tz_convert('US/Pacific')

import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/electric_production.csv', index_col=0)
data.head()

data.tail()

data.index = pd.to_datetime(data.index)

data.columns = ['Energy Production']

data.plot(title="Energy Production Jan 1985--Jan 2018", figsize=(15,6))

rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(data, model='multiplicative')
fig = decomposition.plot()
plt.show()

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

# Train test ssplit

train = data.loc['1985-01-01':'2017-12-01']
test = data.loc['2018-01-01':]

# Train the model

stepwise_model.fit(train)

stepwise_model.fit(train).plot_diagnostics(figsize=(15, 12))
plt.show()

future_forecast = stepwise_model.predict(n_periods=54)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
pd.concat([test,future_forecast],axis=1).plot()

pd.concat([data,future_forecast],axis=1).plot()