import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

resorts = pd.read_csv('resorts.csv', encoding = 'latin-1')
df = pd.DataFrame(resorts)

df['Child friendly'] = df['Child friendly'].map({'Yes': 1, 'No': 0})
df['Snowparks'] = df['Snowparks'].map({'Yes': 1, 'No': 0})
df['Nightskiing'] = df['Nightskiing'].map({'Yes': 1, 'No': 0})
df['Delta altitude'] = df['Highest point'] - df['Lowest point']

skipass_priceDF = df[['Resort', 'Price', 'Total slopes', 'Total lifts', 'Child friendly', 'Snowparks', 'Nightskiing', 'Delta altitude']]

train_data = skipass_priceDF[:450]
test_data = skipass_priceDF[450:]

y = train_data['Price']
features = ['Total slopes', 'Total lifts', 'Child friendly', 'Snowparks', 'Nightskiing', 'Delta altitude']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators = 150, max_depth = 10, random_state = 1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'Resort': test_data.Resort, 'Price': predictions})
print(output)
