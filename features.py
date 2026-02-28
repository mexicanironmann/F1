import pandas as pd

df = pd.read_csv("C:/Users/Owner/Desktop/F1/F1_Dataset.csv")

#print(df.head())
#print(df.tail())
#print(df.columns)
#print(df.dtypes)


##Type conversion 
float_features = ["fp2_FastestLap", "Q1_Time", "Q2_Time", "Q3_Time", "Race_FastestLap"]
int_features = ["Quali_Position", "Race_Position", "fp2_TotalLaps", "Race_TotalLaps", "Race_Points"]
categorical_features = ["GP_Name", "Driver", "Race_Status"]


for item in float_features:
    df[item] = pd.to_timedelta(df[item], errors="coerce").dt.total_seconds()

for item in int_features:
    df[item] = df[item].round().astype('Int64')

for item in categorical_features:
    df[item] = df[item].astype("category")

#print(df.dtypes)

##Feature Selection

#print(df.columns)
#print(df[["fp2_AvgLap", "Quali_Position"]])

df.drop(columns=["fp2_AvgLap", "Quali_Position"], axis=1, inplace=True)
#print(df.columns)

df["Top3"] = (df["Race_Position"] <= 3).astype("Int64")
print(df["Top3"])

#df.to_csv("C:/Users/Owner/Desktop/F1/F1_Dataset.csv", index=False)