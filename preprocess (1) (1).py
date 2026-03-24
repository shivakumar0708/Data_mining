# 1. Import Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# 2. Load the Excel Datasets
file1 = "/mnt/data/Traffic_Dataset_City_A.xlsx"
file2 = "/mnt/data/Traffic_Dataset_City_B.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Add City label if not present
df1["City"] = "City A"
df2["City"] = "City B"

# 3. Combine Both Datasets
data = pd.concat([df1, df2], ignore_index=True)
print("Initial Data Shape:", data.shape)

# 4. Handle Missing Values
data.fillna({
    "Traffic_Volume": data["Traffic_Volume"].mean(),
    "Temperature_C": data["Temperature_C"].mean(),
    "Rain_mm": 0
}, inplace=True)

# 5. Date & Time Processing
data["Date"] = pd.to_datetime(data["Date"])
data["Hour"] = pd.to_datetime(data["Time"], format="%H:%M").dt.hour
data["Day"] = data["Date"].dt.day
data["Month"] = data["Date"].dt.month
data["Weekday"] = data["Date"].dt.weekday

# Weekend Feature
data["Is_Weekend"] = data["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

# 6. Encode Categorical Columns
encoder = LabelEncoder()

data["Day_Type"] = encoder.fit_transform(data["Day_Type"])
data["Weather_Condition"] = encoder.fit_transform(data["Weather_Condition"])
data["City"] = encoder.fit_transform(data["City"])

# 7. Feature Scaling (Normalization)
scaler = MinMaxScaler()

numerical_cols = ["Traffic_Volume", "Temperature_C", "Rain_mm"]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# 8. Drop Unnecessary Column
data.drop(columns=["Time"], inplace=True)

# 9. Save Preprocessed Data
data.to_excel("Preprocessed_Traffic_Data.xlsx", index=False)

print("Preprocessing completed successfully!")
print("Final Data Shape:", data.shape)

# ======================================================
# =============== DATA VISUALIZATION ===================
# ======================================================

# Reload original combined data for graphs (before scaling)
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)
df1["City"] = "City A"
df2["City"] = "City B"

data = pd.concat([df1, df2], ignore_index=True)

data["Date"] = pd.to_datetime(data["Date"])
data["Hour"] = pd.to_datetime(data["Time"], format="%H:%M").dt.hour

# GRAPH 1: Traffic Volume vs Date
plt.figure()
plt.plot(data["Date"], data["Traffic_Volume"])
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.title("Traffic Volume Over Time")
plt.show()

# GRAPH 2: Average Traffic by Hour
hourly_traffic = data.groupby("Hour")["Traffic_Volume"].mean()

plt.figure()
plt.plot(hourly_traffic.index, hourly_traffic.values)
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.title("Average Traffic Volume by Hour")
plt.show()

# GRAPH 3: Weather vs Traffic Volume
weather_traffic = data.groupby("Weather_Condition")["Traffic_Volume"].mean()

plt.figure()
plt.bar(weather_traffic.index, weather_traffic.values)
plt.xlabel("Weather Condition")
plt.ylabel("Average Traffic Volume")
plt.title("Traffic Volume vs Weather Condition")
plt.show()

# GRAPH 4: City-wise Traffic Comparison
city_traffic = data.groupby("City")["Traffic_Volume"].mean()

plt.figure()
plt.bar(city_traffic.index, city_traffic.values)
plt.xlabel("City")
plt.ylabel("Average Traffic Volume")
plt.title("City-wise Traffic Volume Comparison")
plt.show()

print("Graphs generated successfully!")
