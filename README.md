# 📈 MCP_TL Price Forecasting Using LSTM

This project performs time series analysis and forecasting of electricity market prices (MCP_TL) using historical data and LSTM neural networks. It includes data preprocessing, feature engineering, visualization, and model training using TensorFlow/Keras.


---

## 📦 Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
tensorflow
scikit-learn
```

---

## 📁 Dataset

The dataset used in this project is `data_final.csv`, which contains historical hourly market clearing prices and timestamps.

---

## 🔧 Steps Performed

### 1. Data Loading and Preprocessing
- Parsing datetime
- Creating temporal features (hour, day, month, year, weekday)
- Handling missing values
- Exploratory Data Analysis (EDA)
- Correlation matrix

### 2. Visualization
- Distribution plots of MCP_TL, MCP_USD, MCP_EUR
- Time series line plots
- Boxplots for hourly and monthly seasonality

### 3. Feature Engineering
- Lag features: MCP_TL at previous hours (1h, 2h, 3h, 6h, 12h, 24h)
- Output saved as `data_final_processed.csv`

### 4. LSTM Model
- Sequence generation using a rolling 24-hour window
- Data normalization using `MinMaxScaler`
- Model architecture:
  - 1 LSTM layer with 64 units
  - Dropout layer (0.2)
  - Dense output layer
- Early stopping used to prevent overfitting
- Evaluation using MAE and RMSE
- Actual vs predicted plots

---

## 📊 Results

After training, the model's performance is evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

Predicted values closely match actual MCP_TL prices in the test set.

---

## 💾 Outputs

- `data_final_processed.csv`: Preprocessed data
- `lstm_mcp_forecast_model.h5`: Saved LSTM model (uncomment final lines to save)
- Actual vs predicted plot

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

2. Run the script using Jupyter Notebook or any Python IDE.

3. Save the trained model:

```python
model.save("lstm_mcp_forecast_model.h5")
```

---

## 📁 File Structure

```
📦 MCP_Forecasting_Project
 ┣ 📄 data_final.csv
 ┣ 📄 data_final_processed.csv
 ┣ 📄 lstm_mcp_forecast_model.h5
 ┣ 📜 main_forecasting_script.py (optional)
 ┗ 📄 README.md
```

---

## 📌 Notes

- `%matplotlib inline` is for Jupyter Notebook compatibility.
- Make sure the dataset CSV file is in the working directory.
- You may modify the sequence window or model architecture for further improvements.

---

## 🔗 References

- TensorFlow: https://www.tensorflow.org/
- Seaborn Docs: https://seaborn.pydata.org/
- LSTM in Keras: https://keras.io/api/layers/recurrent_layers/lstm/
