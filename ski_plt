import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

resorts = pd.read_csv('resorts.csv', encoding = 'latin-1')

#filter_continent = resorts[resorts.Continent == 'Oceania']
#print(filter_continent[['Country', 'Resort', 'Season']])

fig = plt.figure(figsize=(20,5))

df = pd.DataFrame(resorts)
sns.countplot(x = df['Continent'])

plt.xlabel("Continents")
plt.ylabel("Amount of ski resorts")
plt.title("Ski resorts per continent")
plt.show()
