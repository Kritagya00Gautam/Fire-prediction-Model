from model import eT, RelativeHumidity, VPD, p
from model import best_model
import pandas

data = pandas.read_csv("Global Weather Data.csv")
nepal_data = data[data['Country'] == 'Nepal'].copy()
pokhara_data = nepal_data[nepal_data['City'] == 'Pokhara'].copy()

# here you will send the new data to the model 
# and get the prediction

def predict_fire_risk(new_data, city_weather, model, feature_cols):
    if isinstance(new_data, dict):
        upcoming_df = pandas.DataFrame([new_data])
    else:
        upcoming_df = new_data.copy()
        
    if isinstance(city_weather, dict):
        city_df = pandas.DataFrame([city_weather])
    else:
        city_df = city_weather.copy()
        
   
    upcoming_df['Temperature'] = pandas.to_numeric(upcoming_df['Temperature'], errors='coerce')
    upcoming_df['Air_Density'] = pandas.to_numeric(upcoming_df['Air_Density'], errors='coerce')
    upcoming_df['Soil_Moisture'] = pandas.to_numeric(upcoming_df['Soil_Moisture'], errors='coerce').astype(int)
    
    city_df['Temperature_K'] = pandas.to_numeric(city_df['Temperature_K'], errors='coerce')
    city_df['humidity'] = pandas.to_numeric(city_df['humidity'], errors='coerce')
    
 
    upcoming_df['Relative Humidity'] = RelativeHumidity(city_df['Temperature_K'], city_df['humidity'], upcoming_df['Temperature'])
    upcoming_df['VPD'] = VPD(city_df['Temperature_K'], city_df['humidity'], upcoming_df['Temperature'])
    upcoming_df['atmospheric pressure proxy'] = p(upcoming_df['Air_Density'], upcoming_df['Temperature'])
    
   
    X_new = upcoming_df[feature_cols]
    
   
    preds = model.predict(X_new)
    probs = model.predict_proba(X_new)
    
    results = []
    for i in range(len(preds)):
        results.append({
            'prediction': int(preds[i]),
            'no_fire_prob': float(probs[i][0]),
            'fire_prob': float(probs[i][1])
        })
        
    return results[0] if isinstance(new_data, dict) else results

"""

now you will have to store the new data in a dataframe and then send it to the model for prediction.
since its just for pokhara i have fetched the data from global weathers dataset

ending function is simply 

output = predict_fire_risk(
    new_data=upcoming_reading,
    city_weather=pokhara_baseline,
    model=best_model,
    feature_cols=feature_columns
)

print("Prediction Output:", output)
where pokhara baseline is the baseline data for pokhara and upcoming_reading is the new data you want to predict on.
"""
#