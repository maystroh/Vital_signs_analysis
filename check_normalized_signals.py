import pandas as pd

import matplotlib.pyplot as plt

df_normalized = pd.read_csv('normalized_x.csv', sep=',')

df = pd.read_csv('x.csv', sep=',')

for index, row in df_normalized.iterrows():

    plt.figure(index)
    x_test = list(range(0,len(row)))
    plt.plot(x_test, row)
    plt.ylabel('Signal')
    plt.xlabel('Time')

    plt.figure(index+100)
    plt.plot(x_test, df.iloc[index])
    plt.ylabel('Signal')
    plt.xlabel('Time')

    plt.show()