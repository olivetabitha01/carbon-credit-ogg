import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/carbon_data.csv')

X = df[['Temperature', 'Humidity', 'Sunlight', 'CO2_Emitted_Cars', 'CO2_Absorbed_Trees', 'O2_Released_Trees']]
y = df['Carbon_Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print('Linear Regression R2:', r2_score(y_test, lr_pred))
joblib.dump(lr_model, 'models/linear_regression_model.joblib')

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print('Random Forest R2:', r2_score(y_test, rf_pred))
joblib.dump(rf_model, 'models/random_forest_model.joblib')

print('Training complete. Models saved in models/ folder.')
