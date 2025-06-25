import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import webbrowser
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Chargement des données
df = pd.read_csv("2016_Building_Energy_Benchmarking.csv")

# Aperçu des données
print(df.head())
print(df.describe())
print(df.shape)
print(df.info())
print(df.isna().sum())

# Visualisation des valeurs manquantes
msno.bar(df)
plt.show()
msno.matrix(df)
plt.show()

# Nettoyage
df = df.drop_duplicates()

cols_to_drop = [
    'SecondLargestPropertyUseType',
    'SecondLargestPropertyUseTypeGFA',
    'ThirdLargestPropertyUseType',
    'ThirdLargestPropertyUseTypeGFA',
    'Outlier',
    'YearsENERGYSTARCertified',
    'Comments'
]
data = df.drop(columns=cols_to_drop)

data.drop(columns=["DataYear", "City", "State", "OSEBuildingID"], inplace=True)

# Corrélation
quanti = data.select_dtypes(include=['int64', 'float64'])
corr_matrix = quanti.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de corrélation des variables quantitatives')
plt.show()

cols_corrélées = [
    "Electricity(kWh)", "NaturalGas(therms)",
    "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)", "SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)",
    "GHGEmissionsIntensity", "SiteEnergyUse(kBtu)",
    "ENERGYSTARScore", "BuildingType", "LargestPropertyUseType",
    "CouncilDistrictCode", "Latitude", "Longitude",
    "DefaultData", "ComplianceStatus", "PropertyGFAParking"
]
data.drop(columns=cols_corrélées, inplace=True)

cols_to_drop = [
    'PropertyName', 'Address', 'ZipCode',
    'TaxParcelIdentificationNumber', 'ListOfAllPropertyUseTypes'
]
data.drop(columns=cols_to_drop, inplace=True)

data = data.drop(columns=[
    'SteamUse(kBtu)', 
    'Electricity(kBtu)', 
    'NaturalGas(kBtu)'
])

# Rapport EDA
profile = ProfileReport(data, title="EDA - Données prétraitées", explorative=True)
profile.to_file("eda_pretraitement.html")
print("Rapport EDA généré : eda_pretraitement.html")
webbrowser.open("eda_pretraitement.html")

# Encodage
ohe = OneHotEncoder(use_cat_names=True)
ohe.fit(data)
df_ohe = ohe.transform(data)
df_ohe = df_ohe.dropna()

# Séparation des données
y = df_ohe[['TotalGHGEmissions', 'SiteEnergyUseWN(kBtu)']]
X = df_ohe.drop(columns=['TotalGHGEmissions', 'SiteEnergyUseWN(kBtu)'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèles
model_lr = MultiOutputRegressor(LinearRegression())
model_rf = MultiOutputRegressor(RandomForestRegressor(random_state=42))

model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

y_true_co2 = y_test['TotalGHGEmissions']
y_true_energy = y_test['SiteEnergyUseWN(kBtu)']

def eval_model(y_true_co2, y_true_energy, y_pred, model_name):
    y_pred_co2 = y_pred[:, 0]
    y_pred_energy = y_pred[:, 1]

    r2_co2 = r2_score(y_true_co2, y_pred_co2)
    mae_co2 = mean_absolute_error(y_true_co2, y_pred_co2)

    r2_energy = r2_score(y_true_energy, y_pred_energy)
    mae_energy = mean_absolute_error(y_true_energy, y_pred_energy)

    print(f"Évaluation {model_name}")
    print(f" - CO2 : R² = {r2_co2:.4f}, MAE = {mae_co2:.2f}")
    print(f" - Énergie : R² = {r2_energy:.4f}, MAE = {mae_energy:.2f}\n")

# Évaluation
eval_model(y_true_co2, y_true_energy, y_pred_lr, "LinearRegression")
eval_model(y_true_co2, y_true_energy, y_pred_rf, "RandomForest")
