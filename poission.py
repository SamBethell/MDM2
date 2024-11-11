import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('UKdeaths.csv')

#print(df)
deaths = list(df.iloc[:, 1])
bites = list(df.iloc[:, 2])
bites = [float(i) for i in bites]
population = list(df.iloc[-13:, 3])
population = [float(i) for i in population]
predeaths = deaths[:len(deaths)-1]
death_div_pop = [deaths[-13+i]/population[i] for i in range(13)]


def get_mean(data):
    return sum(data)/len(data)


def create_poission(mean):
    x = [i for i in range(30)]
    y = [((np.e**(-mean))*(mean**i))/math.factorial(i) for i in x]
    return x, y

def get_poission_prob(x, y, value):
    return sum(y[value:])
    
def poission_maker():
    x, y = create_poission(get_mean(predeaths[-10:]))
    print(get_poission_prob(x, y, deaths[-1]))
    plt.title("Poisson Distribution of Fatal Dog Attacks UK (2013-2022 Data)")
    plt.xlabel("Number of Fatal Dog Attacks")
    plt.ylabel("Probability")
    plt.bar(x, y)
    plt.plot([deaths[-1], deaths[-1], deaths[-1]], [0, 0.2, 0.22], linestyle='dashed', color='r', label='2023 Outlier')
    plt.legend()
    plt.savefig('ukfatalpoission.pdf')

def graphic_maker():
    years = [i+2011 for i in range(13)]
    fig, ax1 = plt.subplots()
    plt.title('Growth of Dog Population, Bites, and Fatal Attacks (2011-2023)')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Deaths, Population')
    ax1.plot(years, deaths[-13:], color='r', label='Deaths')
    ax1.plot(years, population[-13:], color='g', label='Population (in Millions)')
    ax1.plot(years, bites[-13:], color='b', label='Bites')
    ax1.set_ylim(0,17)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    ax2.set_ylabel('Bites')  # we already handled the x-label with ax1
    ax2.plot(years, bites[-13:], color='b', label='Bites')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0,10000)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    plt.savefig('ukstatsgraphic.pdf')
    plt.show()

poission_maker()
graphic_maker()
