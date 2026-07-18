An advanced Machine Learning repository dedicated to predicting fire outbreaks and wildfire risks. By analyzing environmental variables, weather patterns, and historical data, this model provides actionable insights to mitigate hazards and optimize emergency response.

---

## 📌 Features

*   **Multi-Model Evaluation:** Implements various algorithms (e.g., Random Forest, XGBoost, Logistic Regression) to ensure optimal accuracy.
*   **Feature Engineering:** Advanced preprocessing pipeline handling meteorological data (Temperature, Humidity, Wind Speed, Rainfall).
*   **Data Visualization:** Exploratory Data Analysis (EDA) notebooks highlighting critical correlations and wildfire triggers.
*   **Scalable Architecture:** Modular code structure designed for easy integration into web dashboards or API endpoints.

---

## 📊 Dataset & Features

The model utilizes environmental attributes typically sourced from meteorological and satellite observations:

| Feature Name | Description | Unit |
| :--- | :--- | :--- |
| `Temperature` | Ambient air temperature | °C |
| `RH` | Relative Humidity | % |
| `Ws` | Wind Speed | km/h |
| `Rain` | Total daily rainfall | mm |
| `FFMC / DMC / DC` | Fine Fuel Moisture Code / Duff Moisture Code / Drought Code | Index |
| `Classes` | Target variable indicating the presence or absence of fire | boolean values cause it will automatically change it|

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed. 

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Kritagya00Gautam/Fire-prediction-Model.git](https://github.com/Kritagya00Gautam/Fire-prediction-Model.git)
   cd Fire-prediction-Model
