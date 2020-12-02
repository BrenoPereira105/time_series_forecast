import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from fbprophet import Prophet

#here we choose the style for the plot 
matplotlib.style.use('ggplot')

# Import as Dataframe
df = pd.read_csv("random_data.csv")

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=10)

forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

m.plot(forecast)
plt.show()


