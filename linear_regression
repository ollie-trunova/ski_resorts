import numpy as np
import pandas as pd

resorts = pd.read_csv('resorts.csv', encoding = 'latin-1')

df = pd.DataFrame(resorts)

df['Child friendly'] = df['Child friendly'].map({'Yes': 1, 'No': 0})
df['Snowparks'] = df['Snowparks'].map({'Yes': 1, 'No': 0})
df['Nightskiing'] = df['Nightskiing'].map({'Yes': 1, 'No': 0})
df['Delta altitude'] = df['Highest point'] - df['Lowest point']

pass_priceDF = df[['Resort', 'Price', 'Total slopes', 'Total lifts', 'Child friendly', 'Snowparks', 'Nightskiing', 'Delta altitude']]

def rmse(targets, predictions):
    return np.sqrt((np.square(predictions - targets)).mean())

def calculateWeights(x_train, y_train):
    w = np.linalg.inv(np.transpose(x_train).dot(x_train)).dot(np.transpose(x_train).dot(y_train))
    return w

x_train = pass_priceDF.iloc[:300, 2:]
y_train = pass_priceDF.iloc[:300, 1]
x_test = pass_priceDF.iloc[300:, 2:]
y_test = pass_priceDF.iloc[300:, 1]

w = calculateWeights(x_train, y_train)

# training error
y_train_pred = x_train.dot(w)
train_error = rmse(y_train, y_train_pred)
print('Train error: ', train_error)

#testing error
y_test_pred = x_test.dot(w)
test_error = rmse(y_test, y_test_pred)
print('Test error: ', test_error)

# x1 = totalslopes, x2 = totallifts, x3 = childfriendly, x4 = snowpark, x5 = nightskiing, x6 = deltaaltitude
new_pred = np.array([150, 40, 1, 1, 0, 1500]).dot(w)
print('Predicted price: ', new_pred)

