import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


df = pd.read_csv('UKdeaths.csv')
deaths = list(df.iloc[:, 1])
print(deaths)
bites = list(df.iloc[:, 2])
bites = [float(i) for i in bites]
population = list(df.iloc[-13:, 3])
population = [float(i) for i in population]
predeaths = deaths[:len(deaths)-1]
death_div_pop = [deaths[-13+i]/population[i] for i in range(13)]

x = df.iloc[:, 0]

fig, ax = plt.subplots()
ax.plot(x, deaths)
plt.title("Fatal dog attacks, UK 1981 - 2023")
plt.xlabel("Year")
plt.ylabel("Deaths")
plt.grid()
plt.savefig("ukfatalgraph.pdf")
plt.show()