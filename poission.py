import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('UKdeaths.csv')

#print(df)
deaths = list(df.iloc[:, 1])
attacks = list(df.iloc[:, 2])
population = list(df.iloc[-13:, 3])
population = [float(i) for i in population]
print(population)
predeaths = deaths[:len(deaths)-1]
death_div_pop = [deaths[-13+i]/population[i] for i in range(13)]
print(death_div_pop)
def get_mean(data):
    return sum(data)/len(data)


def create_poission(mean):
    x = [i for i in range(30)]
    y = [((np.e**(-mean))*(mean**i))/math.factorial(i) for i in x]
    return x, y

def get_poission_prob(x, y, value):
    return sum(y[value:])
    

x, y = create_poission(get_mean(predeaths[-10:]))
print(get_poission_prob(x, y, deaths[-1]))
plt.title("Poission Distribution of Fatal Dog Attacks UK (2013-2022 Data)")
plt.xlabel("Number of Fatal Dog Attacks")
plt.ylabel("Probability")
plt.bar(x, y)
plt.plot([deaths[-1], deaths[-1], deaths[-1]], [0, 0.2, 0.22], linestyle='dashed', color='r', label='2023 Outlier')
plt.legend()
plt.savefig('ukfatalpoission.pdf')
