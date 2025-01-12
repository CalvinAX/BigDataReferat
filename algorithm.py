import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime

# Daten laden
produktdaten = pd.read_csv('./data/real_product_data.csv')
wetterdaten = pd.read_csv('./data/real_weather_data.csv')

produktdaten["DATUM"] = pd.to_datetime(produktdaten["DATUM"]).dt.strftime("%Y-%m-%d")

produktdaten = produktdaten.groupby('DATUM', as_index=False).agg({
    'PREIS': 'mean'  # Durchschnitt der Preise pro Tag
})

# Daten zusammenführen
daten = pd.merge(produktdaten, wetterdaten, on='DATUM')

daten['MONTH'] = pd.to_datetime(daten['DATUM']).dt.month
daten['DAY_OF_YEAR'] = pd.to_datetime(daten['DATUM']).dt.dayofyear

daten['PRICE_ROLLED'] = daten['PREIS'].rolling(window=7).mean().fillna(daten['PREIS'])

# Wochentag und Saison als kategorische Features
daten['WEEKDAY'] = pd.to_datetime(daten['DATUM']).dt.dayofweek
daten['SEASON'] = pd.to_datetime(daten['DATUM']).dt.quarter

daten['PREV_DAY_PRICE'] = daten['PREIS'].shift(1)  # Add previous day's price
daten['PRICE_CHANGE'] = daten['PREIS'].diff()  # Add price change

print(daten)

# Features aktualisieren
X = daten[['DURCHSCHNITT_TEMP_DE', 'MONTH', 'DAY_OF_YEAR', 'WEEKDAY', 'SEASON', 'PREV_DAY_PRICE', 'PRICE_CHANGE']]
y = daten['PREIS'] # -> R2 0.96

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelltraining

### Lineare Regression
#model = LinearRegression()
#model.fit(X_train, y_train)

### Random Forest Regressor
# Hyperparameter-Grid definieren
# Einstellungen für das finden von besten Einstellungen für den RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search durchführen
# Gibt das Modell mit den besten Einstellungen wieder
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Bestes Modell verwenden
model = grid_search.best_estimator_
print(model.get_params())


# Modellbewertung
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
print(f"Prediction: {y_pred}")

# 1. Vorhergesagte vs. Tatsächliche Werte
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Tatsächliche Werte')
plt.ylabel('Vorhergesagte Werte')
plt.title('Vorhergesage vs. Tatsächliche Werte')
plt.tight_layout()
plt.grid()
plt.show()

"""# 2. Residuenplot
residuen = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuen, alpha=0.5)
plt.xlabel('Vorhergesagte Werte')
plt.ylabel('Residuen')
plt.title('Residuenplot')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.grid()
plt.show()"""

"""# 3. Feature Importance Plot (nur für Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()"""

"""# 4. Korrelations Matrix
numeric_col = daten.select_dtypes(include=["number"])
#numeric_col = numeric_col.dropna()
corr_matrix = numeric_col.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap der Korrelationen")
plt.show()"""

