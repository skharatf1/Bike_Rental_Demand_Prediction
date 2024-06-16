import pandas as pd
data = pd.read_csv('MumbaiBikeData.csv', encoding = 'ISO-8859-1')

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek


data = pd.get_dummies(data, columns=['Seasons', 'Holiday', 'Functioning Day'], drop_first=True)

from sklearn.preprocessing import StandardScaler

numerical_features = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']

scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.histplot(data['Rented Bike Count'], bins = 30, kde = True)
plt.title('Distribution of Rented Bike Count')
plt.xlabel('Rented Bike Count')
plt.ylabel('Frequency')
plt.show()


correlation_matrix = data.corr()
plt.figure(figsize=(14,10))
sns.heatmap(correlation_matrix, annot = True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


from sklearn.model_selection import train_test_split

X = data.drop(['Rented Bike Count', 'Date'], axis = 1)
y = data['Rented Bike Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = { 
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Performance:')
    print(f' RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}')
    print(f' MAE: {mean_absolute_error(y_test, y_pred):.2f}')
    print(f' R^2: {r2_score(y_test, y_pred):.2f}')
    print()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_:.2f}')


best_model = grid_search.best_estimator_
feature_importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()