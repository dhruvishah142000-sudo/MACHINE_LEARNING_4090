import pandas as pd

winedatac=pd.read_csv('D:/Winter 2023/wine.csv')

print(winedatac)
print(winedatac.head())
print("shape\n\n\n",winedatac.shape)
print("columns\n\n\n",winedatac.columns)
print("dtypes\n\n\n",winedatac.dtypes)
print("ndim\n\n\n",winedatac.ndim)
print("size\n\n\n",winedatac.size)


winedatae=pd.read_excel('D:/Winter 2023/wine.xlsx')

print('\n')

