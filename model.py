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