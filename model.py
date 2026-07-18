import pandas 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pandas.read_csv("Global Weather Data.csv")
predicting_data = pandas.read_csv("synthetic_fire_dataset copy.csv")

nepal_data = data[data['Country'] == 'Nepal'].copy()
pokhara_data = nepal_data[nepal_data['City'] == 'Pokhara'].copy()

def eT(temprature):
    down = temprature + 273.3
    up = 17.27 *temprature
    return 0.61078 * np.exp(up/down)

def RelativeHumidity(city_temp_k, city_humidity, local_temp):
    city_temp_k = pandas.to_numeric(city_temp_k, errors='coerce')
    city_humidity = pandas.to_numeric(city_humidity, errors='coerce')
    local_temp = pandas.to_numeric(local_temp, errors='coerce')

    invalid_counts = (
        city_temp_k.isna().sum(),
        city_humidity.isna().sum(),
        local_temp.isna().sum()
    )
    if any(invalid_counts):
        print('WARNING: RelativeHumidity inputs contain NaNs:', invalid_counts)

    c_temp = city_temp_k.values - 273.15
    c_hum = city_humidity.values
    l_temp = local_temp.values
    e_city = eT(c_temp)
    e = e_city * (c_hum / 100)
    e_local = eT(l_temp)
    rh = (e / e_local) * 100
    return pandas.Series(rh, index=getattr(local_temp, 'index', None))

def VPD(city_temp_k, city_humidity, local_temp):
    city_temp_k = pandas.to_numeric(city_temp_k, errors='coerce')
    city_humidity = pandas.to_numeric(city_humidity, errors='coerce')
    local_temp = pandas.to_numeric(local_temp, errors='coerce')

    invalid_counts = (
        city_temp_k.isna().sum(),
        city_humidity.isna().sum(),
        local_temp.isna().sum()
    )
    if any(invalid_counts):
        print('WARNING: VPD inputs contain NaNs:', invalid_counts)
    c_temp = city_temp_k.values - 273.15
    c_hum = city_humidity.values
    l_temp = local_temp.values
    e_city = eT(c_temp)
    e = e_city * (c_hum / 100)
    e_local = eT(l_temp)
    
    return (e_local - e)

def p(density, temperatures):
    density = pandas.to_numeric(density, errors='coerce')
    temperatures = pandas.to_numeric(temperatures, errors='coerce')

    def count_nans(value):
        try:
            return pandas.isna(value).sum()
        except Exception:
            return int(pandas.isna(value))

    invalid_counts = (
        count_nans(density),
        count_nans(temperatures)
    )
    if any(invalid_counts):
        print('WARNING: p inputs contain NaNs:', invalid_counts)

    return density * 287.058 * (temperatures + 273.15)

predicting_data = predicting_data.reset_index(drop=True)
nepal_data = nepal_data.reset_index(drop=True)
pokhara_data = pokhara_data.reset_index(drop=True)

pokhara_data['Temperature_K'] = pandas.to_numeric(pokhara_data['temperature'], errors='coerce') + 273.15
pokhara_data['humidity'] = pandas.to_numeric(pokhara_data['humidity'], errors='coerce')
predicting_data['Temperature'] = pandas.to_numeric(predicting_data['Temperature'], errors='coerce')

predicting_data['Relative Humidity'] = RelativeHumidity(
    pokhara_data['Temperature_K'], 
    pokhara_data['humidity'], 
    predicting_data['Temperature']
)
predicting_data['VPD'] = VPD(
    pokhara_data['Temperature_K'], 
    pokhara_data['humidity'], 
    predicting_data['Temperature']
)
predicting_data['atmospheric pressure proxy'] = p(
    predicting_data['Air_Density'],
    predicting_data['Temperature']
)

predicting_data['Soil_Moisture'] = predicting_data['Soil_Moisture'].astype(int)
predicting_data['Fire_Detection'] = predicting_data['Fire_Detection'].astype(int)

engineered_feature_columns = [
                   'Temperature', 
                   'Air_Density', 
                   'Relative Humidity', 
                   'VPD', 
                   'Soil_Moisture', 
                   'atmospheric pressure proxy'
                   ]

X = predicting_data[engineered_feature_columns]
y = predicting_data['Fire_Detection']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameters_dictionary = {
    "criterion": ["gini", "entropy"],
    "max_depth": [4, 6, 8, 12],          
    "min_samples_split": [5, 10, 15],    
    "min_samples_leaf": [2, 4, 8],      
    "max_features": ["sqrt", "log2"],   
    "n_estimators": [100, 150, 200]      
}

randomized_grid_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=parameters_dictionary,
    n_iter=15,
    scoring="accuracy",                  
    cv=5,
    n_jobs=-1,
    random_state=42
)

randomized_grid_search.fit(X_train, y_train)
best_model = randomized_grid_search.best_estimator_

train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)

print(f"Train Set Accuracy Score: {train_accuracy:.2%}")
print(f"Test Set Accuracy Score:  {test_accuracy:.2%}")

gap = train_accuracy - test_accuracy
if gap <= 0.05:
    print("SUCCESS: The model is properly regularized and generalizes well to unseen data.")
else:
    print("WARNING: Minor gap remains. Consider increasing min_samples_leaf further.")