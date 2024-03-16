import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# from neuralforecast.core import NeuralForecast
# from neuralforecast.models import NHITS, NBEATS #, PatchTST
#
# from neuralforecast.losses.numpy import mae, mse

from nixtlats import TimeGPT

with open("E:\yyb\cliport-master//timegpt_api_key.txt", 'r') as file:
    API_KEY = file.read()

df = pd.read_csv('C://Users//Beta//Desktop//medium_views_published_holidays.csv')
df['ds'] = pd.to_datetime(df['ds'])

df.head()

published_dates = df[df['published'] == 1]

fig, ax = plt.subplots(figsize=(12,8))

ax.plot(df['ds'], df['y'])
ax.scatter(published_dates['ds'], published_dates['y'], marker='o', color='red', label='New article')
ax.set_xlabel('Day')
ax.set_ylabel('Total views')
ax.legend(loc='best')

fig.autofmt_xdate()


plt.tight_layout()
plt.show()

train = df[:-168]
test = df[-168:]

future_exog = test[['unique_id', 'ds', 'published', 'is_holiday']]

timegpt = TimeGPT(token=API_KEY)

timegpt_preds = []

for i in range(0, 162, 7):
    timegpt_preds_df = timegpt.forecast(
        df=df.iloc[:1213 + i],
        X_df=future_exog[i:i + 7],
        h=7,
        finetune_steps=10,
        id_col='unique_id',
        time_col='ds',
        target_col='y'
    )

    preds = timegpt_preds_df['TimeGPT']

    timegpt_preds.extend(preds)
test['TimeGPT'] = timegpt_preds

test.head()