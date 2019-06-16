import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('kc_house_data.csv')

plt.figure()
df.plot.scatter(x='sqft_living', y='price')
plt.title('Зависимость цены от жилой площади')

plt.figure()
df.plot.scatter(x='sqft_lot', y='price')
plt.title('Зависимость цены от общей площади')

plt.figure()
df.groupby('yr_built')['price'].mean().plot(x='yr_built', y='price')
plt.title('Средняя цена дома, построенного в таком-то году')

print('Статистические характеристики цены:')
print('Средняя цена дома:', df['price'].mean())
print('Среднее квадратическое отклонение:', df['price'].std())
print('Максимальная цена за дом:', df['price'].max())
print('Минимальная цена за дом:', df['price'].min())

plt.show()