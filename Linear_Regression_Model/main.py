import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Data prep
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x=lifesat[["GDP per capita (USD)"]].values
y=lifesat[["Life satisfaction"]].values


#Data visualisation
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()


# Select a linear model
model = LinearRegression()

# Train the model
model.fit(x, y)

ax = lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])

x_line = np.linspace(23_500, 62_500, 200)
y_line = model.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, color='red', linewidth=2, label='Linear fit')
ax.legend()

plt.show()

# Make a new prediction for cyprus
x_new = [[37_655.2]] # Cyprus GDP per capita in 2020
print(model.predict(x_new))