
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from scipy import stats
 
 
## ## PART 1: EXPLORATORY DATA ANALYSIS (EDA)


# STEP 1: LOADING THE DATA
## Importing the dataset into the environment

df = pd.read_csv(
    "C:\\Users\\rarju\\OneDrive\\Desktop\\Python CSE375\\Air Quality analyses\\AirQualityUCI.csv",
    sep=";",
    decimal=",",
    na_values=-200
)

print("Original Data Frames Shape :-", '\n', df.shape)
print("\nFirst 5 rows of the data :-", '\n', df.head())



# STEP 2: INITIAL INSPECTION OF DATA
## Checking columns, data types and missing values

print("\nColoumns in the dataset :-", '\n', df.columns.tolist())
print("\nData types of each coloumn :-", '\n', df.dtypes)
print("\nChecking for missing values :-", '\n', df.isnull().sum())



# STEP 3: DATA CLEANING
## Dropping unnecessary empty coloumns and rows

df.drop(columns=["Unnamed: 15", "Unnamed: 16"], inplace=True, errors="ignore")
df.dropna(how="all", inplace=True)

print("\nShape after dropping empty rows/cols :-", '\n', df.shape)



# STEP 4: DATA TYPE CONVERSION
## Converting Date and Time into a single DateTime index

df["DateTime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S"
)
df.set_index("DateTime", inplace=True)
df.drop(columns=["Date", "Time"], inplace=True)

print("\nUpdated Data Types :-", '\n', df.dtypes)



# STEP 5: FEATURE CREATION
## Extracting Hour and Month from the index

df["Hour"]  = df.index.hour
df["Month"] = df.index.month

print("\nNewly created features (Hour/Month) :-", '\n', df[["Hour", "Month"]].head(10))



# STEP 6: HANDLEING THE MISSING VALUES
## Missing values are filled using median for numerical data

print("\nMissing values before filling :-", '\n', df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after filling :-", '\n', df.isnull().sum())



# STEP 7: RENAMING THE COLUMNS
## Making coloumn names more readable and short

df.rename(columns={
    "CO(GT)"        : "CO",
    "PT08.S1(CO)"   : "Sensor_CO",
    "NMHC(GT)"      : "NMHC",
    "C6H6(GT)"      : "C6H6",
    "PT08.S2(NMHC)" : "Sensor_NMHC",
    "NOx(GT)"       : "NOx",
    "PT08.S3(NOx)"  : "Sensor_NOx",
    "NO2(GT)"       : "NO2",
    "PT08.S4(NO2)"  : "Sensor_NO2",
    "PT08.S5(O3)"   : "Sensor_O3",
    "T"             : "Temperature",
    "RH"            : "Humidity",
    "AH"            : "AbsHumidity"
}, inplace=True)

print("\nNew coloumn names :-", '\n', df.columns.tolist())



# STEP 8: SUMMARY STATISTICS
## Statistical overview of the numerical data

print("\nSummary Statistics for variables :-", '\n', df.describe())



# STEP 9: CORRELATION MATRIX
## Checking the relationship between variables

corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix :-", '\n', corr)
print(corr)
print("\nTop correlations with CO level :-", '\n', corr["CO"].sort_values(ascending=False))



# STEP 10: OUTLIER DETECTION
## ## IQR method - used to handle the outliers

key_cols = ["CO", "C6H6", "NOx", "NO2", "Temperature", "Humidity"]

for col in key_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: IQR={IQR:.2f}, Outliers Detected={len(outliers)}")

print("\nEDA Completed. Transitioning to Objectives Analysis...\n")

 
 
# ============================================================
#  ## ## PART 2: ANALYSIS — 10 OBJECTIVES


month_names = {3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
               7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct",
               11:"Nov", 12:"Dec", 1:"Jan", 2:"Feb"}



# OBJECTIVE 1: OVERALL CO LEVEL TREND
## Monitoring the average CO levels over time

daily_co = df["CO"].resample("D").mean()

print("\nOBJECTIVE 1: CO Level Trend Statistics :-", '\n', daily_co.describe())

plt.figure(figsize=(12, 4))
plt.plot(daily_co.index, daily_co.values, color="steelblue")
plt.title("Objective 1: Daily Average CO Level Over Time")
plt.xlabel("Date")
plt.ylabel("CO (mg/m³)")
plt.tight_layout()
plt.show()



# OBJECTIVE 2: SENSOR CORRELATION INSIGHT
## Checking how different sensors are related to each other using Seaborn Heatmap

sensor_cols = ["Sensor_CO", "Sensor_NMHC", "Sensor_NOx", "Sensor_NO2", "Sensor_O3"]
sensor_corr = df[sensor_cols].corr()

print("\nOBJECTIVE 2: Sensor Correlation Matrix :-", '\n', sensor_corr)

plt.figure(figsize=(8, 6))
sns.heatmap(sensor_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Objective 2: Sensor Correlation Heatmap (Seaborn)")
plt.tight_layout()
plt.show()



# OBJECTIVE 3: BENZENE (C6H6) BEHAVIOR
## Analyzing Benzene concentration and its relation with other pollutants

print("\nOBJECTIVE 3: Benzene (C6H6) Statistical Summary :-", '\n', df["C6H6"].describe())

for col in ["CO", "NOx", "NO2"]:
    print(f"Correlation: C6H6 vs {col} :- {df['C6H6'].corr(df[col]):.4f}")

plt.figure(figsize=(8, 4))
sns.scatterplot(x="CO", y="C6H6", data=df, alpha=0.3, s=10, color="tomato")
plt.title("Objective 3: Benzene (C6H6) vs CO")
plt.xlabel("CO (mg/m³)")
plt.ylabel("C6H6 (µg/m³)")
plt.tight_layout()
plt.show()



# OBJECTIVE 4: NOx AND NO2 POLLUTION PATTERNS
## Comparing daily trends of NOx and NO2 pollutants

daily_nox = df["NOx"].resample("D").mean()
daily_no2 = df["NO2"].resample("D").mean()

print("\nOBJECTIVE 4: Correlation NOx vs NO2 :-", '\n', df['NOx'].corr(df['NO2']))

plt.figure(figsize=(12, 4))
sns.lineplot(data=daily_nox, label="NOx", color="purple")
sns.lineplot(data=daily_no2, label="NO2", color="green")
plt.title("Objective 4: Daily Average NOx and NO2")
plt.xlabel("Date")
plt.ylabel("Concentration")
plt.legend()
plt.tight_layout()
plt.show()



# OBJECTIVE 5: TEMPERATURE AND HUMIDITY INFLUENCE
## Investigating how environmental factors affect CO levels using Seaborn

print("\nOBJECTIVE 5: Environmental Influence on CO :-")
print(f"Correlation: CO vs Temperature :- {df['Temperature'].corr(df['CO']):.4f}")
print(f"Correlation: CO vs Humidity    :- {df['Humidity'].corr(df['CO']):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(x="Temperature", y="CO", data=df, ax=axes[0], alpha=0.2, s=5, color="coral")
axes[0].set_title("Temperature vs CO")

sns.scatterplot(x="Humidity", y="CO", data=df, ax=axes[1], alpha=0.2, s=5, color="teal")
axes[1].set_title("Humidity vs CO")

plt.suptitle("Objective 5: Temperature and Humidity Influence")
plt.tight_layout()
plt.show()



# OBJECTIVE 6: TIME-BASED (HOURLY) TRENDS
## Analyzing morning and evening peak pollution hours using Seaborn Barplot

hourly_avg = df.groupby("Hour")[["CO", "NOx", "NO2", "C6H6"]].mean().reset_index()

print("\nOBJECTIVE 6: Hourly Average Pollution :-", '\n', hourly_avg)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
pollutants = ["CO", "NOx", "NO2", "C6H6"]
palettes   = ["Blues", "Purples", "Greens", "Oranges"]

for i, (pol, pal) in enumerate(zip(pollutants, palettes)):
    ax = axes[i // 2][i % 2]
    sns.barplot(x="Hour", y=pol, data=hourly_avg, ax=ax, palette=pal, hue="Hour", legend=False)
    ax.set_title(f"Hourly Avg: {pol}")
    ax.set_xticks(range(0, 24))

plt.suptitle("Objective 6: Hourly Pollution Trends")
plt.tight_layout()
plt.show()



# OBJECTIVE 7: SEASONAL / MONTHLY VARIATION
## Identifying monthly patterns in air quality using Seaborn

monthly_avg = df.groupby("Month")[["CO", "NOx", "NO2", "Temperature"]].mean().reset_index()
monthly_avg["MonthName"] = monthly_avg["Month"].map(month_names)

print("\nOBJECTIVE 7: Monthly Average Trends :-", '\n', monthly_avg)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
pollutants_seasonal = ["CO", "NOx", "NO2", "Temperature"]
palettes_seasonal   = ["coolwarm", "viridis", "magma", "YlOrRd"]

for i, (col, pal) in enumerate(zip(pollutants_seasonal, palettes_seasonal)):
    ax = axes[i // 2][i % 2]
    sns.barplot(x="MonthName", y=col, data=monthly_avg, ax=ax, palette=pal, hue="MonthName", legend=False)
    ax.set_title(f"Monthly Avg: {col}")

plt.suptitle("Objective 7: Monthly / Seasonal Variation")
plt.tight_layout()
plt.show()




# OBJECTIVE 8: SIMPLE LINEAR REGRESSION (SLR)
## ## SLR method - used to predict a dependent variable based on an independent variable

# Independent Variable (X): Sensor_CO, Dependent Variable (y): CO
X = df[["Sensor_CO"]]
y = df["CO"]

# Splitting data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initializing and training the model
slr_model = LinearRegression()
slr_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = slr_model.predict(X_test)

# Calculating evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nOBJECTIVE 8: Simple Linear Regression Results :-")
print(f"Mean Squared Error (MSE) :- {mse:.4f}")
print(f"R-squared Score (R2)      :- {r2:.4f}")
print(f"Model Intercept           :- {slr_model.intercept_:.4f}")
print(f"Coefficient (Slope)       :- {slr_model.coef_[0]:.4f}")



# OBJECTIVE 9: REGRESSION VISUALIZATION
## Plotting the training and testing data against the regression line

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.3, s=5, color="gray", label="Training Data")
plt.scatter(X_test, y_test, alpha=0.5, s=5, color="blue", label="Testing Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")

plt.title("Objective 9: SLR - Predicting CO from Sensor_CO (Train vs Test)")
plt.xlabel("Sensor_CO (PT08.S1)")
plt.ylabel("CO (mg/m³)")
plt.legend()
plt.tight_layout()
plt.show()



# OBJECTIVE 10: HYPOTHESIS TESTING
## ## T-Test method - used to compare the means of two groups

# Null Hypothesis (H0): Mean CO level at 9 AM is same as 2 PM
# Alternative Hypothesis (H1): Mean CO levels are significantly different

group_morning = df[df["Hour"] == 9]["CO"]
group_afternoon = df[df["Hour"] == 14]["CO"]

t_stat, p_val = stats.ttest_ind(group_morning, group_afternoon)

print("\nOBJECTIVE 10: Hypothesis Testing (T-Test) Results :-")
print(f"Morning Mean CO :- {group_morning.mean():.4f}")
print(f"Afternoon Mean CO :- {group_afternoon.mean():.4f}")
print(f"T-statistic value :- {t_stat:.4f}")
print(f"P-value obtained  :- {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("\nConclusion :- Reject Null Hypothesis (H0). Significant difference detected between Morning and Afternoon.")
else:
    print("\nConclusion :- Fail to Reject Null Hypothesis (H0). No significant difference detected.")




 